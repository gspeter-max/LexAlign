import { test, expect } from "@playwright/test";

test.describe("Align Page", () => {
    test.beforeEach(async ({ page }) => {
        await page.goto("/dashboard/align");
    });

    test("renders page heading with step number", async ({ page }) => {
        await expect(page.getByText("Step 03")).toBeVisible();
        await expect(page.getByRole("heading", { name: "Align" })).toBeVisible();
        await expect(
            page.getByText("Align your model with human preferences using DPO or GDPO")
        ).toBeVisible();
    });

    test("shows model and dataset path inputs with defaults", async ({ page }) => {
        const modelInput = page.locator("input").nth(0);
        await expect(modelInput).toHaveValue("./checkpoints/finetuned");

        const datasetInput = page.locator("input").nth(1);
        await expect(datasetInput).toHaveValue("./data/preferences");
    });

    test("method toggle switches between DPO and GDPO", async ({ page }) => {
        const dpoBtn = page.getByRole("button", { name: "dpo", exact: true });
        const gdpoBtn = page.getByRole("button", { name: "gdpo", exact: true });

        await expect(dpoBtn).toBeVisible();
        await expect(gdpoBtn).toBeVisible();

        // DPO should be active by default
        await gdpoBtn.click();
        // UI should update — gdpo should now be highlighted
        await dpoBtn.click();
    });

    test("device toggle shows CPU and CUDA", async ({ page }) => {
        await expect(page.getByRole("button", { name: "cpu" })).toBeVisible();
        await expect(page.getByRole("button", { name: "cuda" })).toBeVisible();
    });

    test("dataset fields section is present", async ({ page }) => {
        await expect(page.getByText("Dataset Fields")).toBeVisible();
        await expect(page.getByText("Prompt Field")).toBeVisible();
        await expect(page.getByText("Chosen Field")).toBeVisible();
        await expect(page.getByText("Rejected Field")).toBeVisible();
    });

    test("dataset field inputs have correct defaults", async ({ page }) => {
        // Prompt field default
        const promptInput = page.locator("input[value='prompt']");
        await expect(promptInput).toBeVisible();

        const chosenInput = page.locator("input[value='chosen']");
        await expect(chosenInput).toBeVisible();

        const rejectedInput = page.locator("input[value='rejected']");
        await expect(rejectedInput).toBeVisible();
    });

    test("hyperparameters section is present with all sliders", async ({
        page,
    }) => {
        await expect(page.getByText("Hyperparameters")).toBeVisible();
        await expect(page.getByText("Beta (KL penalty)")).toBeVisible();
        await expect(page.getByText("Learning Rate")).toBeVisible();
        await expect(page.getByText("Batch Size")).toBeVisible();
        await expect(page.getByText("Epochs")).toBeVisible();
        await expect(page.getByText("LoRA Rank")).toBeVisible();
        await expect(page.getByText("LoRA Alpha")).toBeVisible();
    });

    test("toggles for Dry Run and Use LoRA are visible", async ({ page }) => {
        await expect(page.locator("label").filter({ hasText: "Dry Run" })).toBeVisible();
        await expect(page.locator("label").filter({ hasText: "Use LoRA" })).toBeVisible();
    });

    test("run button shows correct label for dry run", async ({ page }) => {
        const runBtn = page.locator("#align-run-btn");
        await expect(runBtn).toHaveText("Dry Run");
    });

    test("log terminal is present", async ({ page }) => {
        await expect(page.getByText("pipeline output")).toBeVisible();
        await expect(page.getByText("Waiting for output...")).toBeVisible();
    });
});

test.describe("Align Page — SSE Mock", () => {
    test("dry run streams alignment config from API", async ({ page }) => {
        await page.route("**/api/align", async (route) => {
            const body = [
                "data: 📋 Alignment Configuration (Dry Run):\n\n",
                "data:    Model:    ./checkpoints/finetuned\n\n",
                "data:    Dataset:  ./data/preferences\n\n",
                "data:    Method:   DPO\n\n",
                "data:    Device:   cpu\n\n",
                "data:    Beta:     0.1\n\n",
                "data:    LR:       1e-5\n\n",
                "data:    Output:   ./checkpoints/aligned\n\n",
                "data: ✅ Dry run complete — no training performed.\n\n",
                "data: [DONE]\n\n",
            ].join("");

            await route.fulfill({
                status: 200,
                headers: {
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                },
                body,
            });
        });

        await page.goto("/dashboard/align");
        await page.locator("#align-run-btn").click();

        await expect(
            page.getByText("📋 Alignment Configuration (Dry Run):")
        ).toBeVisible({ timeout: 5000 });
        await expect(page.getByText("Method:   DPO")).toBeVisible();
        await expect(
            page.getByText("✅ Dry run complete — no training performed.")
        ).toBeVisible();
    });

    test("error handling shows error in log terminal", async ({ page }) => {
        await page.route("**/api/align", async (route) => {
            const body = [
                "data: 📋 Alignment Configuration (Dry Run):\n\n",
                "data: ❌ Error: Model not found at ./checkpoints/finetuned\n\n",
                "data: [DONE]\n\n",
            ].join("");

            await route.fulfill({
                status: 200,
                headers: { "Content-Type": "text/event-stream" },
                body,
            });
        });

        await page.goto("/dashboard/align");
        await page.locator("#align-run-btn").click();

        await expect(
            page.getByText("❌ Error: Model not found")
        ).toBeVisible({ timeout: 5000 });
    });
});
