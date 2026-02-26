import { test, expect } from "@playwright/test";

test.describe("Landing Page", () => {
    test.beforeEach(async ({ page }) => {
        await page.goto("/");
    });

    test("renders the page with correct title", async ({ page }) => {
        await expect(page).toHaveTitle(/LexAlign/);
    });

    test("navigation bar is visible with logo and links", async ({ page }) => {
        // Logo
        const nav = page.locator("nav");
        await expect(nav).toBeVisible();
        await expect(nav.getByText("LexAlign")).toBeVisible();

        // Launch Dashboard link
        const dashboardLink = nav.getByText("Launch Dashboard");
        await expect(dashboardLink).toBeVisible();

        // Navigation links
        await expect(nav.getByText("Pipeline")).toBeVisible();
        await expect(nav.getByText("Features")).toBeVisible();
        await expect(nav.getByText("Models")).toBeVisible();
    });

    test("hero section renders heading and CTA", async ({ page }) => {
        // "Lex" text
        await expect(page.getByText("Lex", { exact: true })).toBeVisible();
        // "Align" text
        await expect(page.getByText("Align", { exact: true }).first()).toBeVisible();

        // CTA button
        const ctaButton = page.getByText("See the Pipeline");
        await expect(ctaButton).toBeVisible();
    });

    test("hero stats section shows correct values", async ({ page }) => {
        await expect(page.getByText("Training Methods")).toBeVisible();
        await expect(page.getByText("Quantization", { exact: true })).toBeVisible();
        await expect(page.getByText("DPO+GDPO")).toBeVisible();
        await expect(page.getByText("Any LLM")).toBeVisible();
    });

    test("badge shows 'Open Source ML Pipeline'", async ({ page }) => {
        await expect(page.getByText("Open Source ML Pipeline")).toBeVisible();
    });

    test("pipeline section renders with heading", async ({ page }) => {
        const pipelineSection = page.locator("#pipeline");
        await pipelineSection.scrollIntoViewIfNeeded();

        await expect(page.getByText("Three stages.")).toBeVisible();
        await expect(page.getByText("One command.")).toBeVisible();
    });

    test("pipeline section has all three stages", async ({ page }) => {
        const pipelineSection = page.locator("#pipeline");
        await pipelineSection.scrollIntoViewIfNeeded();

        await expect(page.getByText("Step 01")).toBeVisible();
        await expect(page.getByText("Step 02")).toBeVisible();
        await expect(page.getByText("Step 03")).toBeVisible();

        await expect(pipelineSection.getByRole("heading", { name: "Download" })).toBeVisible();
        await expect(pipelineSection.getByRole("heading", { name: "Fine-Tune" })).toBeVisible();
    });

    test("code snippet section shows YAML config", async ({ page }) => {
        const codeBlock = page.locator("pre code");
        await codeBlock.scrollIntoViewIfNeeded();

        await expect(codeBlock).toContainText("model:");
        await expect(codeBlock).toContainText('method: "lora"');
        await expect(codeBlock).toContainText("learning_rate:");
    });

    test("features section renders bento grid heading", async ({ page }) => {
        const featuresSection = page.locator("#features");
        await featuresSection.scrollIntoViewIfNeeded();

        await expect(page.getByText("Capabilities")).toBeVisible();
        await expect(page.getByText("Everything you need to")).toBeVisible();
    });

    test("models section shows marquee chips", async ({ page }) => {
        const modelsSection = page.locator("#models");
        await modelsSection.scrollIntoViewIfNeeded();

        await expect(page.getByText("Works with any HuggingFace model")).toBeVisible();
        // Check at least one model chip
        await expect(page.getByText("GPT-2").first()).toBeVisible();
        await expect(page.getByText("LLaMA 3").first()).toBeVisible();
    });

    test("CTA footer shows pip install command", async ({ page }) => {
        const footer = page.getByText("pip install lexalign");
        await footer.scrollIntoViewIfNeeded();
        await expect(footer).toBeVisible();
    });

    test("footer is present with credits", async ({ page }) => {
        await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
        await expect(page.getByText("Built with Next.js + Magic UI + Aceternity UI")).toBeVisible();
    });

    test("page has dark background and no white flash", async ({ page }) => {
        const bgColor = await page.evaluate(() => {
            return getComputedStyle(document.querySelector("main")!).backgroundColor;
        });
        // Should be near-black, not white
        expect(bgColor).not.toBe("rgb(255, 255, 255)");
    });
});
