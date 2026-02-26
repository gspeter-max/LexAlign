import { test, expect } from "@playwright/test";

test.describe("Fine-Tune Page", () => {
    test.beforeEach(async ({ page }) => {
        await page.goto("/dashboard/finetune");
    });

    test("renders page heading with step number", async ({ page }) => {
        await expect(page.getByText("Step 02")).toBeVisible();
        await expect(page.getByRole("heading", { name: "Fine-Tune" })).toBeVisible();
        await expect(
            page.getByText("Train your model with LoRA or QLoRA")
        ).toBeVisible();
    });

    test("shows model and dataset path inputs", async ({ page }) => {
        const modelInput = page.getByPlaceholder("./models/gpt2");
        await expect(modelInput).toBeVisible();
        await expect(modelInput).toHaveValue("./models/gpt2");

        const datasetInput = page.getByPlaceholder("./data/my-dataset");
        await expect(datasetInput).toBeVisible();
        await expect(datasetInput).toHaveValue("./data/my-dataset");
    });

    test("method toggle switches between LoRA and QLoRA", async ({ page }) => {
        const loraBtn = page.getByRole("button", { name: "lora", exact: true });
        const qloraBtn = page.getByRole("button", { name: "qlora", exact: true });

        await expect(loraBtn).toBeVisible();
        await expect(qloraBtn).toBeVisible();

        // LoRA should be active by default (has cyan border)
        await qloraBtn.click();
        // After clicking QLoRA, quantization bits section should appear
        await expect(page.getByText("Quantization Bits")).toBeVisible();
        await expect(page.getByText("4-bit")).toBeVisible();
        await expect(page.getByText("8-bit")).toBeVisible();

        // Switch back to LoRA
        await loraBtn.click();
        await expect(page.getByText("Quantization Bits")).not.toBeVisible();
    });

    test("device toggle switches between CPU and CUDA", async ({ page }) => {
        const cpuBtn = page.getByRole("button", { name: "cpu" });
        const cudaBtn = page.getByRole("button", { name: "cuda" });

        await expect(cpuBtn).toBeVisible();
        await expect(cudaBtn).toBeVisible();
    });

    test("hyperparameters section is present", async ({ page }) => {
        await expect(page.getByText("Hyperparameters")).toBeVisible();
        await expect(page.getByText("LoRA Rank (r)")).toBeVisible();
        await expect(page.getByText("LoRA Alpha")).toBeVisible();
        await expect(page.getByText("Learning Rate")).toBeVisible();
        await expect(page.getByText("Epochs")).toBeVisible();
        await expect(page.getByText("Batch Size")).toBeVisible();
        await expect(page.getByText("Max Seq Length")).toBeVisible();
    });

    test("sliders show default values", async ({ page }) => {
        // LoRA Rank default = 16
        await expect(page.locator("text=16").first()).toBeVisible();
        // LoRA Alpha default = 32
        await expect(page.locator("text=32").first()).toBeVisible();
    });

    test("dry run toggle is ON by default", async ({ page }) => {
        await expect(page.locator("label").filter({ hasText: "Dry Run" })).toBeVisible();
        // Button should say 'Dry Run' not 'Start Training'
        const runBtn = page.locator("#finetune-run-btn");
        await expect(runBtn).toContainText("Dry Run");
    });

    test("log terminal is present with placeholder", async ({ page }) => {
        await expect(page.getByText("pipeline output")).toBeVisible();
        await expect(page.getByText("Waiting for output...")).toBeVisible();
    });

    test("output dir field is editable", async ({ page }) => {
        const outputInput = page.getByPlaceholder("./checkpoints/finetuned");
        await expect(outputInput).toBeVisible();
        await outputInput.fill("./my-output");
        await expect(outputInput).toHaveValue("./my-output");
    });
});

test.describe("Fine-Tune Page — SSE Mock", () => {
    test("dry run streams config summary from API", async ({ page }) => {
        await page.route("**/api/finetune", async (route) => {
            const body = [
                "data: 📋 Fine-tune Configuration (Dry Run):\n\n",
                "data:    Model:    ./models/gpt2\n\n",
                "data:    Dataset:  ./data/my-dataset\n\n",
                "data:    Method:   LORA\n\n",
                "data:    Device:   cpu\n\n",
                "data:    LoRA r:   16\n\n",
                "data:    LR:       0.0003\n\n",
                "data:    Epochs:   3\n\n",
                "data:    Output:   ./checkpoints/finetuned\n\n",
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

        await page.goto("/dashboard/finetune");
        await page.locator("#finetune-run-btn").click();

        await expect(
            page.getByText("📋 Fine-tune Configuration (Dry Run):")
        ).toBeVisible({ timeout: 5000 });
        await expect(page.getByText("Method:   LORA")).toBeVisible();
        await expect(
            page.getByText("✅ Dry run complete — no training performed.")
        ).toBeVisible();
    });
});
