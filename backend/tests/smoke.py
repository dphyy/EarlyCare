from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from app import main


def main_test() -> None:
    with TemporaryDirectory() as state_dir:
        state_root = Path(state_dir)
        main.STATE_STORAGE_ROOT = state_root
        main.CHECKINS_STATE_PATH = state_root / "checkins.json"
        main.TASKS_STATE_PATH = state_root / "volunteer-tasks.json"

        client = TestClient(main.app)

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["product"] == "EarlyCare"

        scenarios = client.get("/scenarios")
        assert scenarios.status_code == 200
        scenario_ids = {scenario["id"] for scenario in scenarios.json()}
        assert {
            "stable",
            "missed-checkin",
            "parkinsons-watch",
            "post-fall-amber",
            "post-fall-red",
            "chronic-illness",
            "mental-wellbeing",
        }.issubset(scenario_ids)

        red_run = client.post("/scenarios/post-fall-red/run")
        assert red_run.status_code == 200
        red_payload = red_run.json()
        assert red_payload["session"]["riskLevel"] == "Red"
        assert len(red_payload["session"]["categories"]) == 8
        assert any(step["id"] == "emergency-alert" and step["status"] == "Triggered" for step in red_payload["session"]["escalationPlan"])

        missed_run = client.post("/scenarios/missed-checkin/run")
        assert missed_run.status_code == 200
        missed_payload = missed_run.json()
        task = next(task for task in missed_payload["tasks"] if task.get("sourceSessionId") == missed_payload["session"]["id"])
        assert task["priority"] == "Today"

        patched = client.patch(f"/volunteer-tasks/{task['id']}", params={"status": "In progress"})
        assert patched.status_code == 200
        assert patched.json()["status"] == "In progress"

        checkins = client.get("/checkins")
        assert checkins.status_code == 200
        assert any(checkin["scenarioId"] == "post-fall-red" for checkin in checkins.json())

    print("backend smoke ok")


if __name__ == "__main__":
    main_test()
