import { test, expect } from "@playwright/test";

test.describe("Navigation End-to-End", () => {
    test("full flow: landing → dashboard → download → finetune → align → back", async ({
        page,
    }) => {
        // 1. Start at landing page
        await page.goto("/");
        await expect(page).toHaveTitle(/LexAlign/);

        // 2. Navigate to dashboard via the header Pipeline link won't work (# anchor),
        //    so go directly
        await page.goto("/dashboard");
        await expect(page.getByText("Pipeline Dashboard")).toBeVisible();

        // 3. Click Download card
        await page.locator("aside").getByText("Download").click();
        await expect(page).toHaveURL("/dashboard/download");
        await expect(page.getByRole("heading", { name: "Download" })).toBeVisible();

        // 4. Navigate to Fine-Tune via sidebar
        await page.locator("aside").getByText("Fine-Tune").click();
        await expect(page).toHaveURL("/dashboard/finetune");
        await expect(page.getByRole("heading", { name: "Fine-Tune" })).toBeVisible();

        // 5. Navigate to Align via sidebar
        await page.locator("aside").getByRole("link", { name: "Align" }).click();
        await expect(page).toHaveURL("/dashboard/align");
        await expect(page.getByRole("heading", { name: "Align" })).toBeVisible();

        // 6. Go back to overview
        await page.locator("aside").getByText("Overview").click();
        await expect(page).toHaveURL("/dashboard");

        // 7. Back to homepage
        await page.getByText("Back to homepage").click();
        await expect(page).toHaveURL("/");
    });

    test("landing page nav links scroll to correct sections", async ({
        page,
    }) => {
        await page.goto("/");

        // Click Pipeline link
        await page.locator("nav").getByText("Pipeline").click();
        // Should scroll to #pipeline section
        await expect(page.locator("#pipeline")).toBeInViewport({ timeout: 3000 });

        // Click Features link
        await page.locator("nav").getByText("Features").click();
        await expect(page.locator("#features")).toBeInViewport({ timeout: 3000 });

        // Click Models link
        await page.locator("nav").getByText("Models").click();
        await expect(page.locator("#models")).toBeInViewport({ timeout: 3000 });
    });

    test("dashboard sidebar active state highlights current page", async ({
        page,
    }) => {
        await page.goto("/dashboard");

        // Overview should be active (has bg-white/10 class via cn)
        const overviewLink = page.locator("aside").getByText("Overview").locator("..");
        await expect(overviewLink).toHaveClass(/bg-white/);

        // Navigate to download
        await page.locator("aside").getByText("Download").click();
        const downloadLink = page.locator("aside").getByText("Download").locator("..");
        await expect(downloadLink).toHaveClass(/bg-white/);
    });
});

test.describe("Responsive Layout", () => {
    test("landing page renders on mobile viewport", async ({ page }) => {
        await page.setViewportSize({ width: 375, height: 812 });
        await page.goto("/");

        // Hero heading should still be visible
        await expect(page.getByText("Lex", { exact: true })).toBeVisible();

        // Stats grid should stack
        await expect(page.getByText("Training Methods")).toBeVisible();
    });

    test("dashboard renders on tablet viewport", async ({ page }) => {
        await page.setViewportSize({ width: 768, height: 1024 });
        await page.goto("/dashboard");

        // Sidebar and main content should both be visible
        await expect(page.locator("aside")).toBeVisible();
        await expect(page.getByText("Pipeline Dashboard")).toBeVisible();
    });
});
