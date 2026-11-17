// Renders submission/figures/*.mmd to PNG in the same font (DejaVu Sans, matplotlib's default) and
// palette as the matplotlib figures. The font is loaded before mermaid measures any text.
//
//   npm install --no-save @mermaid-js/mermaid-cli@11.4.2      (brings mermaid and puppeteer)
//   PUPPETEER_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
//     node submission/render_diagrams.cjs
const fs = require('fs'), path = require('path');
const puppeteer = require('puppeteer');
const FIG = path.join(__dirname, 'figures');
const TTF = path.join(__dirname, '..', '.venv-video/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf');
const MERMAID = require.resolve('mermaid/dist/mermaid.min.js');

const face = (file, weight, style) => `@font-face{font-family:'DejaVu Sans';font-weight:${weight};font-style:${style};` +
  `src:url(data:font/ttf;base64,${fs.readFileSync(path.join(TTF, file)).toString('base64')}) format('truetype');}`;
const css = face('DejaVuSans.ttf', 'normal', 'normal') + face('DejaVuSans-Bold.ttf', 'bold', 'normal') +
            face('DejaVuSans-Oblique.ttf', 'normal', 'italic');

(async () => {
  const cfg = JSON.parse(fs.readFileSync(path.join(FIG, 'mermaid-config.json'), 'utf8'));
  const browser = await puppeteer.launch({executablePath: process.env.PUPPETEER_EXECUTABLE_PATH, headless: true});
  for (const name of ['pipeline', 'dead_ball']) {
    const page = await browser.newPage();
    await page.setViewport({width: 3000, height: 2000, deviceScaleFactor: 2});
    await page.setContent(`<html><head><style>${css} body{margin:0;background:#fff}</style></head><body>
      <p id="warm" style="font-family:'DejaVu Sans'">a<b>b</b><i>c</i></p><div id="c"></div></body></html>`);
    await page.addScriptTag({path: MERMAID});
    await page.evaluate(async () => { await document.fonts.ready; });
    if (!await page.evaluate(() => document.fonts.check("18px 'DejaVu Sans'"))) throw new Error('font not loaded');
    const src = fs.readFileSync(path.join(FIG, `${name}.mmd`), 'utf8');
    await page.evaluate(async (cfg, src) => {
      mermaid.initialize({startOnLoad: false, ...cfg});
      document.getElementById('c').innerHTML = (await mermaid.render('g', src)).svg;
      document.getElementById('warm').remove();
    }, cfg, src);
    await (await page.$('#c svg')).screenshot({path: path.join(FIG, `${name}.png`)});
    await page.close();
    console.log(`${name}.png`);
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
