"""Optional real-browser demo smoke test (pip install playwright; installed Chrome)."""
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome',headless=True)
    page=browser.new_page(viewport={'width':1512,'height':1050})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://localhost:8501',wait_until='networkidle')
    page.get_by_text('Predictive Maintenance XAI',exact=True).wait_for(timeout=60000)
    page.locator('.js-plotly-plot').first.wait_for(timeout=60000)
    page.wait_for_timeout(1500)
    page.screenshot(path=str(ROOT/'outputs/figures/dashboard_browser.png'),full_page=True)
    page.get_by_role('link',name='Prediction',exact=True).click()
    page.get_by_text('Sample index',exact=True).wait_for()
    # Select the actual saved demonstration case through the dataset selectbox.
    import json
    report=json.loads((ROOT/'outputs/metrics/xai_evaluation.json').read_text())
    page.get_by_role('combobox', name='Sample index', exact=True).click()
    page.get_by_role('combobox', name='Sample index', exact=True).fill(str(report['demo_sample_id']))
    page.get_by_role('option',name=str(report['demo_sample_id']),exact=True).click()
    page.get_by_role('button',name='Predict on this sample',exact=True).click()
    page.get_by_text('Why did the model predict this?',exact=True).wait_for(timeout=60000)
    for title in ['Model Runs','Run Comparison','SHAP Explanation','LIME Explanation','Counterfactual Analysis','Model Benchmark','XAI Evaluation','Robustness','Explanation-Guided Analysis','Decision Support','Technical Details','Dataset']:
        page.get_by_role('link',name=title,exact=True).click()
        page.wait_for_timeout(1500)
        assert page.locator('[data-testid="stException"]').count()==0,page.locator('body').inner_text()
        if title=='Decision Support':
            page.get_by_role('button',name='Apply what-if and re-run prediction',exact=True).click()
            page.get_by_text('After model-level intervention',exact=True).wait_for(timeout=30000)
            page.locator('[data-testid=stMain]').evaluate('(el) => el.scrollTo(0,0)')
            page.wait_for_timeout(500)
            page.screenshot(path=str(ROOT/'outputs/figures/decision_browser.png'),full_page=True)
        print('PASS browser',title,flush=True)
    assert not errors,errors
    browser.close()
