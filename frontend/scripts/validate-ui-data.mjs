import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const dataSource = readFileSync(resolve(here, "../src/data.ts"), "utf8");
const mainSource = readFileSync(resolve(here, "../src/main.tsx"), "utf8");
const typeSource = readFileSync(resolve(here, "../src/types.ts"), "utf8");

const requiredScenarioIds = [
  "stable",
  "missed-checkin",
  "parkinsons-watch",
  "post-fall-amber",
  "post-fall-red",
  "chronic-illness",
  "mental-wellbeing"
];

const requiredCategoryIds = [
  "mental_wellbeing",
  "fall_head_impact",
  "concussion_danger",
  "parkinsons_watch",
  "chronic_illness",
  "medication_food_water",
  "social_isolation",
  "missed_checkin"
];

const requiredUiHooks = [
  "ScenarioRunner",
  "CategoryList",
  "EscalationTrail",
  "updateVolunteerTask",
  "Check-In History",
  "Demo baseline scoring"
];

const missingScenarios = requiredScenarioIds.filter((id) => !dataSource.includes(`id: "${id}"`));
const missingCategories = requiredCategoryIds.filter((id) => !typeSource.includes(`"${id}"`));
const missingHooks = requiredUiHooks.filter((hook) => !mainSource.includes(hook));

if (missingScenarios.length || missingCategories.length || missingHooks.length) {
  console.error(
    JSON.stringify(
      {
        missingScenarios,
        missingCategories,
        missingHooks
      },
      null,
      2
    )
  );
  process.exit(1);
}

console.log("frontend data smoke ok");
