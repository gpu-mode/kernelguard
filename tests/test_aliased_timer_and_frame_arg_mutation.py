"""Regression tests for two AUTO_FILTER evasions closed by this change.

GAP 1 -- aliased/assembled-name timer patch evades TIMER_MONKEYPATCH.
    The dotted-literal form (``torch.cuda.Event.elapsed_time = ...``) is already
    caught textually, but reaching the Event class through ``getattr`` aliases and
    assembling the attribute name from string fragments leaves no literal token.

GAP 2 -- frame-arg indirection evades FRAME_WALK_MUTATION.
    A frame globals/locals dict is passed to a module-local helper and written
    through the bound parameter, which the single-Assign alias tracking misses.

Both repros are empirically clean against upstream main; both should now be
flagged. A precision battery asserts that legitimate kernels stay clean.

Complements PR #277 (gc -> ``__globals__`` subscript route); these are the
aliased-timer and frame-arg-indirection routes.

Stdlib ``unittest`` only -- no new dependencies, matching the repo layout.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kernelguard  # noqa: E402


# --- GAP 1: aliased / assembled-name timer patch (currently clean) ---
GAP1_ALIASED_TIMER = '''
import torch
def custom_kernel(data):
    cu = getattr(torch, "cuda")
    E = getattr(cu, "Ev" + "ent")
    m = "elapsed" + "_time"
    o = getattr(E, m)
    def w(self, other): return o(self, other) / 100.0
    setattr(E, m, w)
    return data
'''

# --- GAP 2: frame-dict passed as an argument, written through a parameter ---
GAP2_FRAME_ARG_MUTATION = '''
import sys
def _mutate(ns, name, fn):
    ns[name] = fn
def wrapper(*a, **k):
    return None
def _i():
    g = sys._getframe(1).f_globals
    _mutate(g, "calculate_stats", wrapper)
def custom_kernel(data):
    _i()
    return data
'''


def analyze(code):
    return kernelguard.analyze_code(code)


def patterns(result):
    return {m["pattern"] for m in result["matched_patterns"]}


class TestAliasedTimerEvasion(unittest.TestCase):
    def test_gap1_now_flagged(self):
        r = analyze(GAP1_ALIASED_TIMER)
        self.assertTrue(r["should_filter"], r["matched_patterns"])
        self.assertEqual(r["classification"], "hacked")
        self.assertIn("TIMER_MONKEYPATCH", patterns(r))

    def test_assignment_form_via_alias_flagged(self):
        # The `<base>.<attr> = ...` write form (not setattr) through an alias.
        code = '''
import torch
def custom_kernel(data):
    cu = getattr(torch, "cuda")
    E = getattr(cu, "Event")
    E.elapsed_time = lambda self, other: 0.0
    return data
'''
        r = analyze(code)
        self.assertTrue(r["should_filter"], r["matched_patterns"])
        self.assertIn("TIMER_MONKEYPATCH", patterns(r))

    def test_setattr_synchronize_on_cuda_alias_flagged(self):
        code = '''
import torch
def custom_kernel(data):
    cu = getattr(torch, "cuda")
    setattr(cu, "synchron" + "ize", lambda: None)
    return data
'''
        r = analyze(code)
        self.assertTrue(r["should_filter"], r["matched_patterns"])
        self.assertIn("TIMER_MONKEYPATCH", patterns(r))


class TestFrameArgMutationEvasion(unittest.TestCase):
    def test_gap2_now_flagged(self):
        r = analyze(GAP2_FRAME_ARG_MUTATION)
        self.assertTrue(r["should_filter"], r["matched_patterns"])
        self.assertEqual(r["classification"], "hacked")
        self.assertIn("FRAME_WALK_MUTATION", patterns(r))

    def test_f_locals_arg_variant_flagged(self):
        code = '''
import sys
def _mutate(ns, name, fn):
    ns[name] = fn
def wrapper(*a, **k):
    return None
def _i():
    g = sys._getframe(1).f_locals
    _mutate(g, "calculate_stats", wrapper)
def custom_kernel(data):
    _i()
    return data
'''
        r = analyze(code)
        self.assertTrue(r["should_filter"], r["matched_patterns"])
        self.assertIn("FRAME_WALK_MUTATION", patterns(r))


class TestPrecisionStaysClean(unittest.TestCase):
    """Clean kernels and legitimate timing must NOT be flagged."""

    def _assert_clean(self, code, forbidden=("TIMER_MONKEYPATCH", "FRAME_WALK_MUTATION")):
        r = analyze(code)
        self.assertFalse(r["should_filter"], (r["classification"], r["matched_patterns"]))
        pats = patterns(r)
        for f in forbidden:
            self.assertNotIn(f, pats, f"unexpected {f}: {r['matched_patterns']}")

    def test_plain_matmul(self):
        self._assert_clean('''
import torch
def custom_kernel(a, b):
    return torch.matmul(a, b)
''')

    def test_genuine_eigh(self):
        self._assert_clean('''
import torch
def custom_kernel(a):
    return torch.linalg.eigh(a)
''')

    def test_legitimate_elapsed_time_use(self):
        # elapsed_time CALLED for timing -- a read/call, never a write.
        self._assert_clean('''
import torch
def custom_kernel(a, b):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = torch.matmul(a, b)
    e.record()
    torch.cuda.synchronize()
    _ = s.elapsed_time(e)
    return out
''')

    def test_getattr_float32(self):
        self._assert_clean('''
import torch
def custom_kernel(a, b):
    dt = getattr(torch, "float32")
    return torch.matmul(a.to(dt), b.to(dt))
''')

    def test_bare_cuda_alias_no_timer_write(self):
        self._assert_clean('''
import torch
def custom_kernel(a, b):
    cu = torch.cuda
    cu.synchronize()
    return torch.matmul(a, b)
''')

    def test_mutate_plain_local_dict(self):
        # Argument is an ordinary local dict, not frame-derived -> no taint.
        self._assert_clean('''
def _mutate(ns, name, fn):
    ns[name] = fn
def custom_kernel(data):
    cfg = {}
    _mutate(cfg, "alpha", 1)
    return data
''')

    def test_reads_f_globals_without_writing(self):
        self._assert_clean('''
import sys
def custom_kernel(data):
    g = sys._getframe(0).f_globals
    _ = len(g)
    return data
''')

    def test_comment_mentioning_timer_patch(self):
        # Comment-only mention of a timer patch must not trip the textual rules.
        self._assert_clean('''
import torch
def custom_kernel(a, b):
    # we do NOT do torch.cuda.Event = FakeEvent or torch.cuda.synchronize = None
    return torch.matmul(a, b)
''')


class TestFrameAccessMutationSplitPreserved(unittest.TestCase):
    """A frame dict passed-but-not-written stays FRAME_WALK_ACCESS telemetry."""

    def test_pass_frame_without_write_stays_access(self):
        code = '''
import sys
def _use(ns):
    return len(ns)
def _i():
    g = sys._getframe(1).f_globals
    return _use(g)
def custom_kernel(data):
    _i()
    return data
'''
        r = analyze(code)
        self.assertFalse(r["should_filter"], r["matched_patterns"])
        pats = patterns(r)
        self.assertIn("FRAME_WALK_ACCESS", pats)
        self.assertNotIn("FRAME_WALK_MUTATION", pats)


class TestExistingRulesUnchanged(unittest.TestCase):
    """Pre-existing positives must remain filtered (no rule weakened)."""

    def test_dotted_event_assign_still_flagged(self):
        code = '''
import torch
class FakeEvent: pass
def custom_kernel(a, b):
    torch.cuda.Event = FakeEvent
    return torch.matmul(a, b)
'''
        r = analyze(code)
        self.assertTrue(r["should_filter"], r["matched_patterns"])
        self.assertIn("TIMER_MONKEYPATCH", patterns(r))

    def test_direct_frame_write_still_flagged(self):
        code = '''
import sys
def wrapper(*a, **k): return None
def custom_kernel(data):
    sys._getframe(1).f_globals["calculate_stats"] = wrapper
    return data
'''
        r = analyze(code)
        self.assertTrue(r["should_filter"], r["matched_patterns"])
        self.assertIn("FRAME_WALK_MUTATION", patterns(r))


if __name__ == "__main__":
    unittest.main()
