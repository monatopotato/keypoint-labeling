// Run with Node and Playwright available on NODE_PATH. No real project data is used.
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');

(async () => {
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const frames = ['detected.jpg', 'empty.jpg', 'broken.jpg', 'already-labeled.jpg'].map((name, i) => ({
      name, status: i === 3 ? 'labeled' : 'unlabeled', people: i === 3 ? 1 : 0, auto_note: ''
    }));
    const people = [{id: 1, source: 'yolo26', keypoints: Array.from({length: 17}, () => [10, 20, 2])}];
    const requests = [];
    let delay = 0;
    await page.route('http://poseframe.test/**', async route => {
      const url = new URL(route.request().url());
      const json = (body, status=200) => route.fulfill({status, contentType: 'application/json', body: JSON.stringify(body)});
      if (url.pathname === '/api/project') return json({folder: 'test', mode: 'path', frames});
      if (url.pathname === '/api/devices') return json({torch_version: '2.14.1+cpu', cuda_available: false});
      if (url.pathname.startsWith('/api/annotations/')) {
        const f = frames.find(f => f.name === url.pathname.split('/').pop());
        return json({people: f.people ? people : [], status: f.status});
      }
      if (url.pathname.startsWith('/api/auto-label/')) {
        assert.equal(route.request().postDataJSON().detection_mode, 'distant');
        assert.equal(route.request().postDataJSON().confidence, 0.15);
        const name = url.pathname.split('/').pop(); requests.push(name);
        if(delay) await new Promise(resolve => setTimeout(resolve, delay));
        const f = frames.find(f => f.name === name);
        if(name === 'broken.jpg') {f.auto_note='Cannot read image'; return json({error: f.auto_note}, 500);}
        f.status = name === 'detected.jpg' ? 'labeled' : 'unlabeled';
        f.people = name === 'detected.jpg' ? 1 : 0;
        f.auto_note = f.people ? '' : 'No confident keypoints detected';
        return json({...f, people: f.people ? people : []});
      }
      if(url.pathname.startsWith('/api/image/')) return route.fulfill({contentType: 'image/jpeg', body: fs.readFileSync(path.join(root,'sample_frames','2026-10-21-18-03-04-000.jpg'))});
      const file = url.pathname === '/' ? 'index.html' : url.pathname.slice(1);
      return route.fulfill({contentType: file.endsWith('.js') ? 'text/javascript' : file.endsWith('.css') ? 'text/css' : 'text/html', body: fs.readFileSync(path.join(root,file))});
    });
    await page.goto('http://poseframe.test/');
    await page.locator('#batchAutoBtn').click();
    await page.waitForFunction(() => document.querySelector('#batchStatus').textContent.startsWith('Finished'));
    await page.waitForFunction(() => document.querySelector('#unlabeledCount').textContent === '2');
    assert.deepEqual(requests, ['detected.jpg','empty.jpg','broken.jpg']);
    assert.equal(await page.locator('[data-filter="unlabeled"]').getAttribute('class'), 'active');
    assert.equal(await page.locator('#frameList .frame-row').count(), 2);
    assert.match(await page.locator('#frameList').innerText(), /No confident keypoints detected/);
    assert.match(await page.locator('#frameList').innerText(), /Cannot read image/);
    requests.length=0;delay=500;
    await page.locator('#batchAutoBtn').click();
    await page.waitForFunction(() => !document.querySelector('#batchStopBtn').classList.contains('hidden'));
    await page.locator('#batchStopBtn').click();
    await page.waitForFunction(() => document.querySelector('#batchStatus').textContent.startsWith('Stopped'));
    assert.ok(requests.length <= 1, 'Stop must prevent the next frame from starting');
    assert.deepEqual(errors, []);
    console.log('Batch browser checks passed: saving results, review filter, errors, skip, stop.');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode=1;});
