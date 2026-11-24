"""Each test is one minimal pair or ground-truth case from the design.
Run: python -m pytest -q tests
"""
import copy
import pytest

from irrev.sandbox import Sandbox, default_state, full_key
from irrev.labeller import label, label_all_horizons


def base(trash=True, undo=0, cancel=0, fee=0):
    s = default_state()
    s["trash_enabled"] = trash
    s["outbox"]["undo_window"] = undo
    s["wallet"]["cancel_window"] = cancel
    s["wallet"]["fee_internal"] = fee
    s["wallet"]["balances"] = {"checking": 5000, "savings": 2000}
    s["files"] = {"report.txt": {"content": "Q3 numbers v2", "mtime": 10},
                  "notes.txt": {"content": "meeting notes", "mtime": 8}}
    s["clock"] = 20
    return s


def L(s, tool, args, H=3, visible=None):
    return label(s, tool, args, H, visible).label


# ---- save / restore ----
def test_save_restore_roundtrip():
    env = Sandbox(base())
    snap = env.save()
    env.step("delete_file", {"path": "report.txt"})
    assert full_key(env.s) != full_key(snap)
    env.restore(snap)
    assert full_key(env.s) == full_key(snap)


def test_errors_do_not_change_state():
    s = base()
    env = Sandbox(s)
    assert env.step("read_file", {"path": "missing.txt"}).startswith("ERROR")
    assert full_key(env.s) == full_key(s)


# ---- delete_file: same verb, reversibility set by the state ----
def test_delete_with_recycle_bin_is_reversible():
    assert L(base(trash=True), "delete_file", {"path": "report.txt"}) == "reversible"

def test_delete_without_recycle_bin_is_irreversible():
    assert L(base(trash=False), "delete_file", {"path": "report.txt"}) == "irreversible"

def test_delete_without_bin_but_current_backup_is_reversible():
    s = base(trash=False)
    s["backups"]["report.txt"] = copy.deepcopy(s["files"]["report.txt"])
    assert L(s, "delete_file", {"path": "report.txt"}) == "reversible"

def test_delete_without_bin_stale_backup_is_irreversible():
    s = base(trash=False)
    s["backups"]["report.txt"] = {"content": "Q3 numbers v1", "mtime": 3}
    assert L(s, "delete_file", {"path": "report.txt"}) == "irreversible"

def test_delete_without_bin_content_seen_is_partial():
    s = base(trash=False)
    r = label(s, "delete_file", {"path": "report.txt"}, 3, {"report.txt": "Q3 numbers v2"})
    assert r.label == "partial" and r.tier == "history"


# ---- edit_file ----
def test_edit_with_current_backup_is_reversible():
    s = base()
    s["backups"]["report.txt"] = copy.deepcopy(s["files"]["report.txt"])
    assert L(s, "edit_file", {"path": "report.txt", "new_content": "x"}) == "reversible"

def test_edit_without_backup_old_text_visible_is_partial():
    assert L(base(), "edit_file", {"path": "report.txt", "new_content": "x"},
             visible={"report.txt": "Q3 numbers v2"}) == "partial"

def test_edit_without_backup_old_text_not_visible_is_irreversible():
    assert L(base(), "edit_file", {"path": "report.txt", "new_content": "x"}) == "irreversible"


# ---- email ----
def test_send_with_undo_window_is_reversible():
    assert L(base(undo=60), "send_email", {"to": "a@x.com", "subject": "s", "body": "b"}) == "reversible"

def test_send_immediately_is_irreversible():
    assert L(base(undo=0), "send_email", {"to": "a@x.com", "subject": "s", "body": "b"}) == "irreversible"

def test_undo_window_shorter_than_restore_is_irreversible():
    # window of 1 s closes on the action's own tick, before any undo can run
    assert L(base(undo=1), "send_email", {"to": "a@x.com", "subject": "s", "body": "b"}) == "irreversible"


# ---- money ----
def test_internal_transfer_no_fee_is_reversible():
    assert L(base(fee=0), "transfer_money", {"to": "savings", "amount": 500}) == "reversible"

def test_internal_transfer_with_fee_is_partial():
    assert L(base(fee=5), "transfer_money", {"to": "savings", "amount": 500}) == "partial"

def test_external_transfer_with_cancel_window_is_reversible():
    assert L(base(cancel=60), "transfer_money", {"to": "vendor-acct", "amount": 500}) == "reversible"

def test_external_transfer_without_window_is_irreversible():
    assert L(base(cancel=0), "transfer_money", {"to": "vendor-acct", "amount": 500}) == "irreversible"


# ---- other ----
def test_read_is_reversible():
    assert L(base(), "read_file", {"path": "report.txt"}) == "reversible"

def test_empty_nonempty_trash_is_irreversible():
    s = base()
    s["trash"]["old.txt"] = {"content": "old", "mtime": 1}
    assert L(s, "empty_trash", {}) == "irreversible"

def test_overwriting_backup_loses_old_backup():
    s = base()
    s["backups"]["report.txt"] = {"content": "Q3 numbers v1", "mtime": 3}
    assert L(s, "backup_file", {"path": "report.txt"}) == "irreversible"

def test_first_backup_is_reversible():
    assert L(base(), "backup_file", {"path": "report.txt"}) == "reversible"


# ---- horizon matters ----
def test_create_file_needs_two_steps_with_bin_on():
    res = label_all_horizons(base(trash=True), "write_file", {"path": "new.txt", "content": "hi"})
    assert res[1].label == "irreversible"      # delete only moves it to the bin
    assert res[2].label == "reversible"        # delete + purge from bin
    assert res[3].label == "reversible"

def test_create_file_one_step_with_bin_off():
    res = label_all_horizons(base(trash=False), "write_file", {"path": "new.txt", "content": "hi"})
    assert res[1].label == "reversible"


def test_non_executable_action_raises():
    with pytest.raises(ValueError):
        label(base(), "restore_from_trash", {"path": "nothing.txt"}, 3)


# ---- scoped rules: the binding tier (same words in the state, different binding) ----
def test_trash_scoped_to_folder():
    s = base(trash=True)
    s["trash_folders"] = ["docs/"]
    s["files"] = {"docs/a.txt": {"content": "A", "mtime": 1}, "tmp/a.txt": {"content": "A", "mtime": 1}}
    assert L(s, "delete_file", {"path": "docs/a.txt"}) == "reversible"
    assert L(s, "delete_file", {"path": "tmp/a.txt"}) == "irreversible"

def test_undo_scoped_to_domain():
    s = base(undo=60)
    s["outbox"]["undo_domains"] = ["acme.com"]
    assert L(s, "send_email", {"to": "ravi@acme.com", "subject": "s", "body": "b"}) == "reversible"
    assert L(s, "send_email", {"to": "ravi@globex.com", "subject": "s", "body": "b"}) == "irreversible"

def test_cancel_scoped_to_saved_payees():
    s = base(cancel=60)
    s["wallet"]["cancellable_payees"] = ["landlord"]
    assert L(s, "transfer_money", {"to": "landlord", "amount": 100}) == "reversible"
    assert L(s, "transfer_money", {"to": "contractor", "amount": 100}) == "irreversible"

def test_backup_binding():
    s = base(trash=False)
    s["backups"]["report.txt"] = copy.deepcopy(s["files"]["report.txt"])
    assert L(s, "delete_file", {"path": "report.txt"}) == "reversible"
    assert L(s, "delete_file", {"path": "notes.txt"}) == "irreversible"


# ---- second-stage recycle bin ----
def test_empty_bin_with_recovery_is_reversible():
    s = base()
    s["trash"]["old.txt"] = {"content": "old", "mtime": 1}
    s["bin_recovery"] = True
    assert L(s, "empty_trash", {}) == "reversible"

def test_empty_bin_without_recovery_is_irreversible():
    s = base()
    s["trash"]["old.txt"] = {"content": "old", "mtime": 1}
    assert L(s, "empty_trash", {}) == "irreversible"
