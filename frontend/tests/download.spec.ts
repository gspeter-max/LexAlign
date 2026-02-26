import { test, expect } from "@playwright/test";

test.describe("Download Page", () => {
    test.beforeEach(async ({ page }) => {
        await page.goto("/dashboard/download");
    });

    test("renders page heading with step number", async ({ page }) => {
        await expect(page.getByText("Step 01")).toBeVisible();
        await expect(page.getByRole("heading", { name: "Download" })).toBeVisible();
        await expect(
            page.getByText("Pull models and datasets from the HuggingFace Hub")
        ).toBeVisible();
    });

    test("shows HuggingFace Token input", async ({ page }) => {
        const tokenInput = page.getByPlaceholder("hf_xxxxxxxxxxxxxxxxxxxx");
        await expect(tokenInput).toBeVisible();
        await expect(tokenInput).toHaveAttribute("type", "password");
    });

    test("shows default model entry with repo field", async ({ page }) => {
        const repoInput = page.getByPlaceholder("repo/model-name");
        await expect(repoInput).toBeVisible();
    });

    test("can add and remove model entries", async ({ page }) => {
        // Initial count of model rows
        const repoInputsBefore = page.getByPlaceholder("repo/model-name");
        const initialCount = await repoInputsBefore.count();

        // Add a model
        await page.getByText("+ Add model").click();
        const repoInputsAfter = page.getByPlaceholder("repo/model-name");
        await expect(repoInputsAfter).toHaveCount(initialCount + 1);

        // Remove one using the × button
        const removeButtons = page.getByText("×");
        await removeButtons.last().click();
        await expect(page.getByPlaceholder("repo/model-name")).toHaveCount(initialCount);
    });

    test("can add and remove dataset entries", async ({ page }) => {
        // Initially no datasets
        await expect(page.getByText("No datasets added")).toBeVisible();

        // Add a dataset
        await page.getByText("+ Add dataset").click();
        await expect(page.getByPlaceholder("repo/dataset-name")).toBeVisible();
        await expect(page.getByText("No datasets added")).not.toBeVisible();

        // Remove it
        const removeButtons = page.locator("button").filter({ hasText: "×" });
        await removeButtons.last().click();
        await expect(page.getByText("No datasets added")).toBeVisible();
    });

    test("toggle switches work (Dry Run, Models Only, Datasets Only)", async ({
        page,
    }) => {
        // Toggles are inside <label> elements with <span> text
        await expect(page.locator("label").filter({ hasText: "Dry Run" })).toBeVisible();
        await expect(page.locator("label").filter({ hasText: "Models Only" })).toBeVisible();
        await expect(page.locator("label").filter({ hasText: "Datasets Only" })).toBeVisible();
    });

    test("run button is disabled without HF token", async ({ page }) => {
        const runBtn = page.locator("#download-run-btn");
        await expect(runBtn).toBeDisabled();
    });

    test("run button enables when HF token is provided", async ({ page }) => {
        await page.getByPlaceholder("hf_xxxxxxxxxxxxxxxxxxxx").fill("hf_test_token_12345");
        const runBtn = page.locator("#download-run-btn");
        await expect(runBtn).toBeEnabled();
    });

    test("log terminal shows 'Waiting for output...'", async ({ page }) => {
        await expect(page.getByText("Waiting for output...")).toBeVisible();
    });

    test("log terminal has terminal header with dots", async ({ page }) => {
        await expect(page.getByText("pipeline output")).toBeVisible();
    });

    test("can fill model repo and output dir", async ({ page }) => {
        const repoInput = page.getByPlaceholder("repo/model-name");
        await repoInput.fill("meta-llama/Llama-3-8B");
        await expect(repoInput).toHaveValue("meta-llama/Llama-3-8B");

        const outputInput = page.getByPlaceholder("./models/name");
        await outputInput.fill("./models/llama3");
        await expect(outputInput).toHaveValue("./models/llama3");
    });
});

test.describe("Download Page — SSE Mock", () => {
    test("dry run streams SSE logs from API", async ({ page }) => {
        // Mock the download API to return a fake SSE stream
        await page.route("**/api/download", async (route) => {
            const body = [
                "data: 🔐 Validating HuggingFace token...\n\n",
                "data: ✅ Authentication successful\n\n",
                "data: ⬇️  Downloading model: gpt2\n\n",
                "data:    [DRY RUN] Would download gpt2 → ./models/gpt2\n\n",
                "data: 🎉 Download complete!\n\n",
                "data: [DONE]\n\n",
            ].join("");

            await route.fulfill({
                status: 200,
                headers: {
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    Connection: "keep-alive",
                },
                body,
            });
        });

        await page.goto("/dashboard/download");

        // Fill token
        await page.getByPlaceholder("hf_xxxxxxxxxxxxxxxxxxxx").fill("hf_test_token");

        // Click run
        await page.locator("#download-run-btn").click();

        // Logs should appear
        await expect(page.getByText("🔐 Validating HuggingFace token...")).toBeVisible({
            timeout: 5000,
        });
        await expect(page.getByText("✅ Authentication successful")).toBeVisible();
        await expect(page.getByText("🎉 Download complete!")).toBeVisible();
    });
});
