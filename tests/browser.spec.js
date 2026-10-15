import { test, expect } from "@playwright/test";
import fs from "node:fs";
const baseUrl = process.env.TEST_BASE_URL || "http://127.0.0.1:5000";

test("real prediction, exact SHAP, evaluation, and responsive layout", async ({
  page,
}) => {
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(baseUrl);
  await expect(
    page.getByText("Held-out accuracy", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".stats")).toContainText("79.37%");
  await expect(page.locator(".stats")).toContainText("59.04%");
  await expect(
    page.getByRole("heading", { name: "Explore a prediction" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Load demo values" }).click();
  const predictionResponse = page.waitForResponse((r) =>
    r.url().endsWith("/api/predict"),
  );
  await page
    .getByRole("button", { name: "Estimate default probability" })
    .click();
  const prediction = await (await predictionResponse).json();
  await expect(
    page.getByText("Estimated next-month default probability"),
  ).toBeVisible();
  const explanationResponse = page.waitForResponse(
    (r) => r.url().endsWith("/api/explain"),
    { timeout: 60000 },
  );
  await page.getByRole("button", { name: "Explain this prediction" }).click();
  const explanation = await (await explanationResponse).json();
  expect(
    Math.abs(
      explanation.explanation.reconstructed_probability -
        prediction.default_probability,
    ),
  ).toBeLessThan(1e-6);
  await expect(
    page.getByText("Background probability:", { exact: false }),
  ).toBeVisible();
  fs.mkdirSync("docs/screenshots", { recursive: true });
  await page.screenshot({
    path: "docs/screenshots/prediction-desktop.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Evaluation", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Evidence behind the estimate" }),
  ).toBeVisible();
  await expect(page.getByText(/Training \/ validation \/ test:/)).toContainText(
    "18000 / 6004 / 5996",
  );
  await expect(
    page.getByRole("heading", { name: "Subgroup audit" }),
  ).toBeVisible();
  await expect
    .poll(
      async () =>
        await page
          .locator("figure img")
          .evaluateAll((imgs) =>
            imgs.every((i) => i.complete && i.naturalWidth > 0),
          ),
    )
    .toBeTruthy();
  await page.screenshot({
    path: "docs/screenshots/evaluation-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: "Prediction", exact: true }).click();
  await expect(
    page.getByLabel("Credit limit (NT$)", { exact: true }),
  ).toHaveValue("200000");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: "docs/screenshots/prediction-mobile.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("input validity and server error are visible", async ({ page }) => {
  await page.goto(baseUrl);
  await page.getByRole("button", { name: "Load demo values" }).click();
  await page.getByLabel("Credit limit (NT$)", { exact: true }).fill("0");
  expect(
    await page
      .getByLabel("Credit limit (NT$)", { exact: true })
      .evaluate((e) => e.validity.valid),
  ).toBeFalsy();
  await page.getByLabel("Credit limit (NT$)", { exact: true }).fill("200000");
  await page.route("**/api/predict", (r) =>
    r.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ error: "Model unavailable for test." }),
    }),
  );
  await page
    .getByRole("button", { name: "Estimate default probability" })
    .click();
  await expect(page.getByRole("alert")).toContainText(
    "Model unavailable for test.",
  );
});
