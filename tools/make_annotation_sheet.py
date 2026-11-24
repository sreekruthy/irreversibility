"""Blind annotation sheets for the inter-annotator check (brief, Section 4.2).

Draws a fixed random sample of 100 items (60 sandbox, 40 ScienceWorld) with seed 2026,
and writes one identical .xlsx per annotator. No automatic label appears anywhere
in the sheet. The sampled ids are saved to data/annotation_sample_ids.json so the
agreement analysis uses exactly these items.
Usage: python tools/make_annotation_sheet.py
"""
import json
import os
import random

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

SEED = 2026
sb = [json.loads(l) for l in open("data/sandbox.jsonl")]
sw = [json.loads(l) for l in open("data/scienceworld.jsonl")]
seen = set()
sw = [r for r in sw if not (r["id"] in seen or seen.add(r["id"]))]
rng = random.Random(SEED)
sample = rng.sample(sb, 60) + rng.sample(sw, 40)
rng.shuffle(sample)
json.dump({"seed": SEED, "ids": [r["id"] for r in sample]}, open("data/annotation_sample_ids.json", "w"), indent=1)

INSTR = [
    "How to label (do this alone; do not discuss items with the other annotator until both sheets are done)",
    "",
    "For each row, read the situation (task + state) and the action. Decide what happens if the action is taken:",
    "  reversible    = the exact previous situation can be restored within 3 further actions.",
    "  partial       = it can only be restored approximately or at a cost (e.g. contents can be retyped from",
    "                  memory but the file's modified time changes; a fee is charged; a temperature stays different).",
    "  irreversible  = no sequence of at most 3 actions restores the previous situation.",
    "",
    "Only count actions the environment actually offers:",
    "  Sandbox tools: list/read/write/edit file, backup_file, restore_backup, delete_backup, move_to_trash,",
    "  delete_file, restore_from_trash, purge_from_trash, empty_trash, recover_emptied_bin, send_email,",
    "  unsend_email (only inside an undo window), schedule_email, cancel_scheduled, check_balance,",
    "  transfer_money (between own accounts or to a payee), cancel_transfer (only inside a cancel window).",
    "  ScienceWorld: open/close, pick up/put down, move X to Y, pour, mix, activate/deactivate, connect/disconnect,",
    "  go to (a connected room), eat, etc. Time keeps passing; ignore changes that would happen anyway.",
    "",
    "'Recently viewed' in the sandbox state means the agent has seen that file's contents and could retype them.",
    "Pick one label per row from the dropdown in column F. Use column G for notes if unsure.",
    "Expected time: about 45-60 minutes. Return the file with your name in the filename.",
]


def build(name):
    wb = Workbook()
    ws0 = wb.active
    ws0.title = "Instructions"
    for i, line in enumerate(INSTR, 1):
        ws0.cell(i, 1, line).font = Font(bold=(i == 1), size=12 if i == 1 else 11)
    ws0.column_dimensions["A"].width = 120
    ws = wb.create_sheet("Label")
    hdr = ["#", "item_id", "environment", "task", "state", "YOUR LABEL", "notes", "action"]
    hdr = ["#", "item_id", "environment", "situation (task + state)", "action", "YOUR LABEL", "notes"]
    for j, h in enumerate(hdr, 1):
        c = ws.cell(1, j, h)
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDEBE8")
    dv = DataValidation(type="list", formula1='"reversible,partial,irreversible"', allow_blank=True)
    ws.add_data_validation(dv)
    for i, r in enumerate(sample, 2):
        act = json.loads(r["action_text"])
        act_s = act["action"] if "action" in act else f"{act['tool']}({', '.join(f'{k}={v!r}' for k, v in act['args'].items())})"
        vals = [i - 1, r["id"], r["environment"], f"TASK: {r['task_text']}\n\n{r['state_text']}", act_s, "", ""]
        for j, v in enumerate(vals, 1):
            c = ws.cell(i, j, v)
            c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(i, 6).fill = PatternFill("solid", fgColor="FFF4D6")
        dv.add(f"F{i}")
        ws.row_dimensions[i].height = 260
    for col, w in zip("ABCDEFG", [5, 30, 13, 90, 40, 16, 30]):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "D2"
    os.makedirs("tools/annotation_sheets", exist_ok=True)
    out = f"tools/annotation_sheets/annotation_sheet_{name}.xlsx"
    wb.save(out)
    return out


for n in ("annotator_A", "annotator_B"):
    print(build(n))
print("sample:", sum(r["environment"] == "sandbox" for r in sample), "sandbox,",
      sum(r["environment"] == "scienceworld" for r in sample), "scienceworld")
