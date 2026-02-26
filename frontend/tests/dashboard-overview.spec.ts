import { test, expect } from "@playwright/test";

test.describe("Dashboard Overview", () => {
    test.beforeEach(async ({ page }) => {
        await page.goto("/dashboard");
    });

    test("renders dashboard layout with sidebar", async ({ page }) => {
        // Sidebar with logo
        const sidebar = page.locator("aside");
        await expect(sidebar).toBeVisible();
        await expect(sidebar.getByText("LexAlign")).toBeVisible();
    });

    test("sidebar has all navigation links", async ({ page }) => {
        const sidebar = page.locator("aside");

        await expect(sidebar.getByRole("link", { name: "Overview" })).toBeVisible();
        await expect(sidebar.getByRole("link", { name: "Download" })).toBeVisible();
        await expect(sidebar.getByRole("link", { name: "Fine-Tune" })).toBeVisible();
        await expect(sidebar.getByRole("link", { name: "Align" })).toBeVisible();
    });

    test("sidebar shows step numbers", async ({ page }) => {
        const sidebar = page.locator("aside");

        await expect(sidebar.getByText("01")).toBeVisible();
        await expect(sidebar.getByText("02")).toBeVisible();
        await expect(sidebar.getByText("03")).toBeVisible();
    });

    test("overview has page heading", async ({ page }) => {
        await expect(page.getByText("Pipeline Dashboard")).toBeVisible();
        await expect(
            page.getByText("Run the full LexAlign pipeline from your browser")
        ).toBeVisible();
    });

    test("overview shows three stage cards", async ({ page }) => {
        const main = page.locator("main");

        // Each card has a unique description
        await expect(main.getByText("Pull models and datasets")).toBeVisible();
        await expect(main.getByText("Train with LoRA or QLoRA")).toBeVisible();
        await expect(main.getByText("Align model to human preferences")).toBeVisible();

        // Three "Open" action links, one per card
        await expect(main.getByText("Open", { exact: true })).toHaveCount(3);
    });

    test("stage cards have Open links", async ({ page }) => {
        const openLinks = page.getByText("Open", { exact: true });
        await expect(openLinks).toHaveCount(3);
    });

    test("sidebar 'Back to homepage' link works", async ({ page }) => {
        const backLink = page.getByText("Back to homepage");
        await expect(backLink).toBeVisible();
        await backLink.click();
        await expect(page).toHaveURL("/");
    });

    test("sidebar navigation to Download works", async ({ page }) => {
        await page.locator("aside").getByText("Download").click();
        await expect(page).toHaveURL("/dashboard/download");
    });

    test("sidebar navigation to Fine-Tune works", async ({ page }) => {
        await page.locator("aside").getByText("Fine-Tune").click();
        await expect(page).toHaveURL("/dashboard/finetune");
    });

    test("sidebar navigation to Align works", async ({ page }) => {
        await page.locator("aside").getByRole("link", { name: "Align" }).click();
        await expect(page).toHaveURL("/dashboard/align");
    });
});
