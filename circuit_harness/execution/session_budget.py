"""Agent-visible server action budgets for Chips public task sessions."""

import json
from pathlib import Path


def budget_snapshot(directory: Path, config: dict) -> dict:
    requests = [
        json.loads(path.read_text()) for path in (directory / "actions").glob("*/request.json")
    ]
    simulations = sum(request["tool"].endswith("_simulate") for request in requests)
    return {
        "max_actions": config["max_actions"],
        "actions_remaining": max(0, config["max_actions"] - len(requests)),
        "max_simulations": config["max_simulations"],
        "simulations_remaining": max(0, config["max_simulations"] - simulations),
        "submission_slot_reserved": True,
    }


def budget_rejection(directory: Path, config: dict, tool: str) -> dict | None:
    if not config.get("budget_feedback"):
        return None
    budget = budget_snapshot(directory, config)
    if budget["actions_remaining"] == 1 and not tool.endswith("_submit"):
        return {"ok": False, "error": "action_reserved_for_submission", "budget": budget}
    return None


def with_budget(reply: dict, directory: Path, config: dict) -> dict:
    if config.get("budget_feedback"):
        return {**reply, "budget": budget_snapshot(directory, config)}
    return reply
