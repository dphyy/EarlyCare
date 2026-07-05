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
        assert health.json()["storage"]["status"] == "ok"
        assert health.json()["storage"]["stateWritable"] is True
        assert health.json()["storage"]["callsWritable"] is True

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

        schedule = client.get("/schedule")
        assert schedule.status_code == 200
        schedule_payload = schedule.json()
        assert len(schedule_payload) == 3
        assert {item["status"] for item in schedule_payload}.issubset({"On track", "Due soon", "Due now", "Overdue"})

        queue = client.get("/operations-queue")
        assert queue.status_code == 200
        queue_payload = queue.json()
        assert len(queue_payload) == 3
        assert [item["queueRank"] for item in queue_payload] == [1, 2, 3]
        assert queue_payload[0]["seniorId"] == "s-001"
        assert queue_payload[0]["priority"] == "Emergency"

        records = client.get("/senior-records")
        assert records.status_code == 200
        records_payload = records.json()
        assert len(records_payload) == 3
        tan_record = next(record for record in records_payload if record["seniorId"] == "s-001")
        assert tan_record["highestRiskLevel"] == "Red"
        assert any(category["id"] == "concussion_danger" for category in tan_record["categories"])

        call_plans = client.get("/call-plans")
        assert call_plans.status_code == 200
        call_plan_payload = call_plans.json()
        assert len(call_plan_payload) == 3
        raman_plan = next(plan for plan in call_plan_payload if plan["seniorId"] == "s-002")
        assert raman_plan["scheduleStatus"] in {"Due now", "Overdue"}
        assert any(question["id"] == "speech-watch" for question in raman_plan["questions"])

        started = client.post("/checkins/start", params={"senior_id": "s-003"})
        assert started.status_code == 200
        assert started.json()["status"] == "In progress"

        completed = client.post(f"/checkins/{started.json()['id']}/complete")
        assert completed.status_code == 200
        assert completed.json()["status"] == "Checked in"

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

        senior_record = client.get(f"/seniors/{missed_payload['session']['seniorId']}/record")
        assert senior_record.status_code == 200
        assert any(event["id"] == missed_payload["session"]["id"] for event in senior_record.json()["timeline"])

        call_plan = client.get(f"/seniors/{missed_payload['session']['seniorId']}/call-plan")
        assert call_plan.status_code == 200
        assert any(question["id"] == "contact-reliability" for question in call_plan.json()["questions"])

    print("backend smoke ok")


if __name__ == "__main__":
    main_test()
