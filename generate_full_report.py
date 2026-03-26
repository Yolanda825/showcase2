#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 requirement.md 生成完整图文结合分析报告  输出: statistics/full_analysis_report.html"""

import pandas as pd, numpy as np, os, re, json, tempfile, ast, hashlib
from collections import Counter
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
# import asyncio, aiohttp  # 已关闭下载，仅引用 report_thumbs 现有文件
from pyecharts import options as opts
from pyecharts.charts import Bar, Scatter, Grid
from pyecharts.globals import JsCode

# ─── 1. 数据 ────────────────────────────────────────────────

def load_data():
    r = pd.read_csv('图像识别_副本/20260313/result2000.csv')
    r = r.dropna(subset=['image_url']); r = r[r['image_url'].astype(str).str.strip() != '']
    a = pd.read_csv('用户是否有下载点击图片_新老用户.csv')
    m = pd.merge(r, a[['device_id','image_url','is_saved','is_download_click','paid_for_url','user_type']],
                 on=['device_id','image_url'], how='inner')
    for c in ['is_saved','is_download_click','paid_for_url']:
        m[c] = m[c].fillna(0).astype(int)
    m['user_type'] = m['user_type'].astype(str).str.strip()
    m.loc[m['user_type'].isin(['\\N','nan','None','']), 'user_type'] = None
    return m

def agg_l2(df):
    g = df.dropna(subset=['scene_l2']).groupby('scene_l2').agg(
        count=('device_id','count'), user_count=('device_id','nunique'),
        click_count=('is_download_click','sum'),
        saved_count=('is_saved','sum'), paid_count=('paid_for_url','sum'),
    ).reset_index()
    total = g['count'].sum()
    g['占比(%)']       = (g['count']/total*100).round(2)
    g['下载点击率(%)'] = (g['click_count']/g['count']*100).round(1)
    g['下载成功率(%)'] = (g['saved_count']/g['click_count'].replace(0,1)*100).round(1)
    g['新增付费率(%)'] = (g['paid_count']/g['click_count'].replace(0,1)*100).round(2)
    g['流失率(%)']     = ((1 - g['saved_count']/g['click_count'].replace(0,1))*100).round(1)
    return g.sort_values('count', ascending=False).reset_index(drop=True)

def display_filter(g):
    """展示过滤：占比>=1% 且排除「未确定」，计算仍用全量"""
    return g[(g['占比(%)'] >= 1) & (g['scene_l2'] != '未确定')].copy()

# ─── 2. 工具 ────────────────────────────────────────────────

def _ch(chart):
    tmp = tempfile.mktemp(suffix='.html'); chart.render(tmp)
    with open(tmp,'r',encoding='utf-8') as f: html = f.read()
    os.unlink(tmp)
    m = re.search(r'<body[^>]*>(.*?)</body>', html, re.DOTALL|re.IGNORECASE)
    return m.group(1).strip() if m else ''

# 缩略图：仅引用 statistics/report_thumbs/ 下已有文件（不下载），便于 GitHub Pages 展示
REPORT_THUMBS_DIR = os.path.join(os.path.dirname(__file__) or '.', 'statistics', 'report_thumbs')

# ---------- 以下为原下载逻辑，已关闭：不再请求网络，直接使用 report_thumbs 中已存在的图 ----------
# REPORT_THUMBS_TIMEOUT = 60
# REPORT_THUMBS_CONCURRENT = 10
# async def _download_thumb_async(session, url, semaphore):
#     """将 url 下载到 statistics/report_thumbs/<hash>.jpg，返回 (url, rel_path 或 None)。"""
#     ...
# def _build_url_to_local(urls):
#     """原：aiohttp 下载后返回 {url: rel_path 或 url}。已改为仅查本地文件。"""
#     return asyncio.run(run())

def _build_url_to_local(urls):
    """仅根据 URL 的 hash 检查 report_thumbs 中是否已有对应文件；有则返回相对路径，无则保留原 URL。不下载。"""
    urls = [u for u in urls if u and str(u).strip().startswith(('http://', 'https://'))]
    urls = list(dict.fromkeys(urls))
    if not urls:
        return {}
    url_to_local = {}
    for u in urls:
        u = str(u).strip()
        key = hashlib.md5(u.encode('utf-8')).hexdigest()[:16]
        rel_path = f"report_thumbs/{key}.jpg"
        abs_path = os.path.join(os.path.dirname(__file__) or '.', 'statistics', 'report_thumbs', f"{key}.jpg")
        if os.path.isfile(abs_path):
            url_to_local[u] = rel_path
        else:
            url_to_local[u] = u  # 本地无图时保留原 URL（GitHub 上可能不显示，但结构不报错）
    return url_to_local

def _norm_url(v):
    """将 image_url 可能出现的 str/tuple/list 规范为字符串 URL，否则返回 ''。"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ''
    if isinstance(v, str):
        s = v.strip()
        if s.startswith(('http://', 'https://')):
            return s
        # 可能是 "(url)" 或 "('url',)" 等被读成字符串
        if s.startswith(('(', '[')):
            try:
                parsed = ast.literal_eval(s)
                return _norm_url(parsed)
            except (ValueError, SyntaxError):
                pass
        return ''
    if isinstance(v, (list, tuple)) and len(v):
        return _norm_url(v[0])
    return ''

def _tbl(df, hl=None, hg=True, n=None):
    if n: df = df.head(n)
    vals = df[hl].tolist() if hl and hl in df.columns else []
    if vals: vmx,vmn = max(vals),min(vals)
    rows = ''
    for _,row in df.iterrows():
        cells = ''
        for col in df.columns:
            v = row[col]; st = ''
            if col == hl and vals:
                norm = (v-vmn)/(vmx-vmn+1e-9)
                gv=int(80+norm*120); rv=int(220-norm*120)
                if not hg: rv,gv = gv,rv
                st = f' style="background:rgba({rv},{gv},100,.25);font-weight:600"'
            if isinstance(v, float): v = f'{v:.1f}'
            cells += f'<td{st}>{v}</td>'
        rows += f'<tr>{cells}</tr>'
    hdr = ''.join(f'<th>{c}</th>' for c in df.columns)
    return f'<div class="tbl-wrap"><table><thead><tr>{hdr}</tr></thead><tbody>{rows}</tbody></table></div>'

def _parse_tags(series):
    result = []
    for x in series.dropna():
        x = str(x).strip()
        if x.startswith('['):
            try: result.extend(ast.literal_eval(x))
            except: pass
        elif x and x != 'nan':
            result.append(x)
    return Counter(result)

def _parse_row_tags(x):
    """解析单行 difficulty_tags，返回标签列表，与 _parse_tags 逻辑一致"""
    if pd.isna(x):
        return []
    x = str(x).strip()
    if x.startswith('['):
        try:
            return list(ast.literal_eval(x))
        except Exception:
            return []
    if x and x != 'nan':
        return [x]
    return []

def _html_wordcloud(counter, title):
    if not counter: return ''
    items = counter.most_common(15); mx = items[0][1]
    colors = ['#24477f','#5a8bc9','#0d6b2c','#b85c00','#5a4a7a','#4a5568','#1a3560','#c41e3a','#0d6b4a']
    tags = ''
    for i,(w,c) in enumerate(items):
        sz = max(15, int(15 + (c/mx)*32)); clr = colors[i % len(colors)]; opa = max(0.55, c/mx)
        tags += f'<span style="font-size:{sz}px;color:{clr};opacity:{opa};margin:5px 10px;display:inline-block;font-weight:700">{w}<sub style="font-size:10px;font-weight:400;opacity:.7">({c})</sub></span>'
    return f'<div style="background:#fff;border-radius:10px;box-shadow:0 2px 6px rgba(0,0,0,.08);padding:16px;text-align:center"><div style="font-size:13px;font-weight:700;color:#374151;margin-bottom:10px">{title}</div>{tags}</div>'

# ─── 3. 图表 ────────────────────────────────────────────────

def chart_s1(g):
    """所有二级场景需求量，横轴=抠图人数"""
    d = g.iloc[::-1]; cats=d['scene_l2'].tolist(); users=d['user_count'].tolist(); shs=d['占比(%)'].tolist()
    h = max(500, len(g)*22)
    return _ch(Bar(init_opts=opts.InitOpts(width="1060px",height=f"{h}px"))
        .add_xaxis(cats).add_yaxis("抠图人数",users,color="#24477f",
            label_opts=opts.LabelOpts(is_show=True,position="right",font_size=10,
                formatter=JsCode("function(p){var s="+json.dumps(shs)+";return p.value+' ('+s[p.dataIndex]+'%)';}")) )
        .reversal_axis().set_global_opts(
            title_opts=opts.TitleOpts(title="全部二级场景需求量",subtitle="括号内为占总需求比例"),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=11)),
            xaxis_opts=opts.AxisOpts(name="抠图人数"),
            tooltip_opts=opts.TooltipOpts(trigger="axis",axis_pointer_type="shadow")))

def chart_s2(gd):
    """展示占比>=1%且非未确定的场景"""
    df = gd.sort_values('下载点击率(%)',ascending=False).reset_index(drop=True)
    good=df[df['下载点击率(%)']>=75].iloc[::-1]; med=df[(df['下载点击率(%)']>=65)&(df['下载点击率(%)']<75)].iloc[::-1]; poor=df[df['下载点击率(%)']<65].iloc[::-1]
    ac = df.iloc[::-1]['scene_l2'].tolist()
    def vf(s):
        m={r['scene_l2']:r['下载点击率(%)'] for _,r in s.iterrows()}; return [m.get(c,None) for c in ac]
    lbl = opts.LabelOpts(is_show=True,position="right",font_size=9,formatter=JsCode("function(p){return p.value!=null?p.value+'%':'';}"))
    h = max(400, len(ac)*26)
    return _ch(Bar(init_opts=opts.InitOpts(width="1060px",height=f"{h}px")).add_xaxis(ac)
        .add_yaxis("效果优 ≥75%",vf(good),color="#0d6b2c",label_opts=lbl)
        .add_yaxis("效果中 65-75%",vf(med),color="#b85c00",label_opts=lbl)
        .add_yaxis("效果待提升 <65%",vf(poor),color="#c41e3a",label_opts=lbl)
        .reversal_axis().set_global_opts(
            title_opts=opts.TitleOpts(title="各二级场景下载点击率（质量认可度）",subtitle="占比≥1%场景 | 🟢≥75% 🟡65-75% 🔴<65%"),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=11)),
            xaxis_opts=opts.AxisOpts(name="下载点击率(%)",max_=110),
            legend_opts=opts.LegendOpts(pos_top="8%"),
            tooltip_opts=opts.TooltipOpts(trigger="axis",axis_pointer_type="shadow")))

def chart_s3(gd):
    df = gd.sort_values('count',ascending=False)
    return _ch(Bar(init_opts=opts.InitOpts(width="1060px",height="440px")).add_xaxis(df['scene_l2'].tolist())
        .add_yaxis("付费率（下载成功占比）",df['下载成功率(%)'].tolist(),color="#24477f",label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("新增付费率",df['新增付费率(%)'].tolist(),color="#5a8bc9",label_opts=opts.LabelOpts(is_show=True,font_size=9,position="top"))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="各二级场景付费率 & 新增付费率",subtitle="占比≥1%场景"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45,font_size=10,overflow="truncate",text_width=70)),
            yaxis_opts=opts.AxisOpts(name="比率(%)",max_=100,axislabel_opts=opts.LabelOpts(formatter="{value}%")),
            legend_opts=opts.LegendOpts(pos_top="8%"),
            tooltip_opts=opts.TooltipOpts(trigger="axis",axis_pointer_type="shadow"),
            datazoom_opts=[opts.DataZoomOpts(type_="inside")]))

def chart_s4(gd):
    df = gd[gd['click_count']>0].sort_values('流失率(%)',ascending=False).iloc[::-1]
    h_d=df[df['流失率(%)']>=50]; m2=df[(df['流失率(%)']>=30)&(df['流失率(%)']<50)]; lo=df[df['流失率(%)']<30]
    ac=df['scene_l2'].tolist()
    def vf(s):
        m={r['scene_l2']:r['流失率(%)'] for _,r in s.iterrows()}; return [m.get(c,None) for c in ac]
    lbl = opts.LabelOpts(is_show=True,position="right",font_size=9,formatter=JsCode("function(p){return p.value!=null?p.value+'%':'';}"))
    h = max(400, len(ac)*26)
    return _ch(Bar(init_opts=opts.InitOpts(width="1060px",height=f"{h}px")).add_xaxis(ac)
        .add_yaxis("高流失 ≥50%",vf(h_d),color="#c41e3a",label_opts=lbl)
        .add_yaxis("中流失 30-50%",vf(m2),color="#b85c00",label_opts=lbl)
        .add_yaxis("低流失 <30%",vf(lo),color="#0d6b2c",label_opts=lbl)
        .reversal_axis().set_global_opts(
            title_opts=opts.TitleOpts(title="各二级场景商业化拦截流失率",subtitle="占比≥1%场景 | 🔴≥50% 🟠30-50% 🟡<30%"),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=11)),
            xaxis_opts=opts.AxisOpts(name="流失率(%)",max_=110),
            legend_opts=opts.LegendOpts(pos_top="8%"),
            tooltip_opts=opts.TooltipOpts(trigger="axis",axis_pointer_type="shadow")))

def chart_s5_subject(df):
    gs = df.dropna(subset=['subject_l1']).groupby('subject_l1').agg(
        count=('device_id','count'),click_count=('is_download_click','sum')).reset_index()
    total = gs['count'].sum()
    gs['需求占比(%)'] = (gs['count']/total*100).round(1)
    gs['下载点击率(%)']=(gs['click_count']/gs['count']*100).round(1)
    gs = gs.sort_values('count',ascending=False).iloc[::-1]
    bar = (Bar(init_opts=opts.InitOpts(width="1060px",height="440px"))
        .add_xaxis(gs['subject_l1'].tolist())
        .add_yaxis("需求占比(%)",gs['需求占比(%)'].tolist(),color="#24477f",label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("下载点击率(%)",gs['下载点击率(%)'].tolist(),color="#5a8bc9",
            label_opts=opts.LabelOpts(position="right",font_size=10,formatter=JsCode("function(p){return p.value+'%';}")))
        .reversal_axis().set_global_opts(
            title_opts=opts.TitleOpts(title="一级主体：需求占比 & 下载点击率",title_textstyle_opts=opts.TextStyleOpts(font_size=16,font_weight="bold")),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=11)),
            legend_opts=opts.LegendOpts(pos_top="8%"),
            tooltip_opts=opts.TooltipOpts(trigger="axis")))
    grid = Grid(init_opts=opts.InitOpts(width="1060px",height="440px")).add(
        bar, grid_opts=opts.GridOpts(pos_left="22%", pos_right="5%"))
    return _ch(grid)

def chart_s5_sub(df, scene):
    sub = df[df['scene_l2']==scene].dropna(subset=['subject_l2'])
    if sub.empty: return ''
    gs = sub.groupby('subject_l2').agg(count=('device_id','count'),click_count=('is_download_click','sum')).reset_index()
    total = gs['count'].sum()
    gs['需求占比(%)'] = (gs['count']/total*100).round(1)
    gs['下载点击率(%)']=(gs['click_count']/gs['count']*100).round(1)
    gs = gs.sort_values('count',ascending=False).head(8).iloc[::-1]
    bar = (Bar(init_opts=opts.InitOpts(width="520px",height="340px"))
        .add_xaxis(gs['subject_l2'].tolist())
        .add_yaxis("需求占比(%)",gs['需求占比(%)'].tolist(),color="#24477f",label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("下载点击率(%)",gs['下载点击率(%)'].tolist(),color="#5a8bc9",
            label_opts=opts.LabelOpts(position="right",font_size=9,formatter=JsCode("function(p){return p.value+'%';}")))
        .reversal_axis().set_global_opts(
            title_opts=opts.TitleOpts(title=f"【{scene}】",
                title_textstyle_opts=opts.TextStyleOpts(font_size=13)),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=9,overflow="break",formatter=JsCode("function(v){return v;}")),name_gap=30),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=9)),
            legend_opts=opts.LegendOpts(pos_top="14%",item_width=10,item_height=8,textstyle_opts=opts.TextStyleOpts(font_size=9)),
            tooltip_opts=opts.TooltipOpts(trigger="axis")))
    grid = Grid(init_opts=opts.InitOpts(width="520px",height="340px")).add(
        bar, grid_opts=opts.GridOpts(pos_left="32%", pos_right="8%"))
    return _ch(grid)

def chart_s5_subject_l2(df):
    """二级主体：需求占比 & 下载点击率，格式同附录一级主体条形图。"""
    gs = df.dropna(subset=['subject_l2']).groupby('subject_l2').agg(
        count=('device_id', 'count'), click_count=('is_download_click', 'sum')).reset_index()
    if gs.empty:
        return ''
    total = gs['count'].sum()
    gs['需求占比(%)'] = (gs['count'] / total * 100).round(1)
    gs['下载点击率(%)'] = (gs['click_count'] / gs['count'].replace(0, 1) * 100).round(1)
    gs = gs.sort_values('count', ascending=False)
    # 限制展示数量并控制高度，避免图表过高导致不渲染或 style 缺少 px
    max_bars = 40
    gs = gs.head(max_bars).iloc[::-1]
    chart_h = max(440, min(len(gs) * 22, 900))
    bar = (Bar(init_opts=opts.InitOpts(width="1060px", height=chart_h))
        .add_xaxis(gs['subject_l2'].tolist())
        .add_yaxis("需求占比(%)", gs['需求占比(%)'].tolist(), color="#24477f", label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("下载点击率(%)", gs['下载点击率(%)'].tolist(), color="#5a8bc9",
            label_opts=opts.LabelOpts(position="right", font_size=10, formatter=JsCode("function(p){return p.value+'%';}")))
        .reversal_axis().set_global_opts(
            title_opts=opts.TitleOpts(title="二级主体：需求占比 & 下载点击率", title_textstyle_opts=opts.TextStyleOpts(font_size=16, font_weight="bold")),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=11)),
            legend_opts=opts.LegendOpts(pos_top="4%"),
            tooltip_opts=opts.TooltipOpts(trigger="axis")))
    grid = Grid(init_opts=opts.InitOpts(width="1060px", height=chart_h)).add(
        bar, grid_opts=opts.GridOpts(pos_left="22%", pos_right="5%", pos_top="14%"))
    html = _ch(grid)
    # 修复 pyecharts 可能输出的 height:NNN 缺少 px 导致图表不展示
    html = re.sub(r'(height:\s*)(\d+)(\s*[;"])', r'\g<1>\g<2>px\g<3>', html)
    return html

# ─── 蝴蝶图 ─────────────────────────────────────────────────

def _butterfly_rate(df, metric_col, title, max_val=100, scene_list=None):
    """scene_list: 指定要展示的场景列表（与其他模块一致的 占比>=1% 场景）"""
    g = df.dropna(subset=['scene_l2','user_type'])
    g = g[g['scene_l2'] != '未确定']
    g = g.groupby(['scene_l2','user_type']).agg(
        count=('device_id','count'),click_count=('is_download_click','sum'),
        saved_count=('is_saved','sum'),paid_count=('paid_for_url','sum')).reset_index()
    g['下载点击率(%)']=(g['click_count']/g['count']*100).round(1)
    g['下载成功率(%)']=(g['saved_count']/g['click_count'].replace(0,1)*100).round(1)
    g['新增付费率(%)']=(g['paid_count']/g['click_count'].replace(0,1)*100).round(2)
    g['流失人数'] = g['click_count'] - g['saved_count']
    g['流失率(%)'] = np.where(g['click_count'] > 0, (1 - g['saved_count']/g['click_count'])*100, 0).round(1)
    for ut in ['新用户','老用户']:
        mask = g['user_type']==ut
        ut_total = g.loc[mask,'count'].sum()
        g.loc[mask,'场景占比(%)'] = (g.loc[mask,'count']/ut_total*100).round(1)
    num_col_map = {'下载点击率(%)':'click_count','下载成功率(%)':'saved_count',
                   '新增付费率(%)':'paid_count','场景占比(%)':'count','流失率(%)':'流失人数'}
    denom_col_map = {'流失率(%)':'click_count','下载成功率(%)':'click_count','新增付费率(%)':'click_count'}
    num_col = num_col_map.get(metric_col, 'count')
    denom_col = denom_col_map.get(metric_col, 'count')
    # 使用传入的 scene_list（占比>=1%场景），按总需求量排序取 TOP
    if scene_list is None:
        total = df[df['scene_l2']!='未确定'].dropna(subset=['scene_l2']).groupby('scene_l2')['device_id'].count()
        scene_list = total.sort_values(ascending=False).head(10).index.tolist()
    nu = g[g['user_type']=='新用户'].set_index('scene_l2')
    ou = g[g['user_type']=='老用户'].set_index('scene_l2')
    scenes = [s for s in scene_list if s in nu.index or s in ou.index]
    if not scenes: return ''
    sr = scenes[::-1]
    nv = [-float(nu.loc[s, metric_col]) if s in nu.index else 0 for s in sr]
    ov = [float(ou.loc[s, metric_col]) if s in ou.index else 0 for s in sr]
    # 绝对值：用单引号构建 JS 对象，避免双引号与 pyecharts 序列化冲突
    is_share = metric_col == '场景占比(%)'
    items = []
    for s in sr:
        nn = int(nu.loc[s, num_col]) if s in nu.index else 0
        nt = int(nu.loc[s, denom_col]) if s in nu.index else 0
        on = int(ou.loc[s, num_col]) if s in ou.index else 0
        ot = int(ou.loc[s, denom_col]) if s in ou.index else 0
        key = s.replace("'", "\\'")
        items.append(f"'{key}':[{nn},{nt},{on},{ot}]")
    abs_js = '{' + ','.join(items) + '}'
    if max_val is None:
        actual = max(max(abs(v) for v in nv), max(ov), 0.1)
        max_val = max(5, int(np.ceil(actual / 5) * 5))
    h = max(420, len(sr) * 36)
    lbl_l = opts.LabelOpts(is_show=True,position="left",font_size=9,
        formatter=JsCode("function(p){var v=Math.abs(p.value);return v>0?v+'%':'';}"))
    lbl_r = opts.LabelOpts(is_show=True,position="right",font_size=9,
        formatter=JsCode("function(p){return p.value>0?p.value+'%':'';}"))
    if is_share:
        tooltip_fn = ("function(p){var m=" + abs_js + ";"
            "var r='<b>'+p[0].name+'</b><br/>';"
            "var d=m[p[0].name]||[0,0,0,0];"
            "p.forEach(function(i,idx){"
            "var lbl=idx===0?'新用户':'老用户';"
            "var v=Math.abs(i.value);"
            "var n=idx===0?d[0]:d[2];"
            "r+=lbl+': '+v.toFixed(1)+'% ('+n+'人)<br/>';});"
            "return r;}")
    else:
        tooltip_fn = ("function(p){var m=" + abs_js + ";"
            "var r='<b>'+p[0].name+'</b><br/>';"
            "var d=m[p[0].name]||[0,0,0,0];"
            "p.forEach(function(i,idx){"
            "var lbl=idx===0?'新用户':'老用户';"
            "var v=Math.abs(i.value);"
            "var n=idx===0?d[0]:d[2];var t=idx===0?d[1]:d[3];"
            "r+=lbl+': '+v.toFixed(1)+'% ('+n+'/'+t+')<br/>';});"
            "return r;}")
    grid_top = 58
    bar = (Bar(init_opts=opts.InitOpts(width="520px",height=f"{h}px"))
        .add_xaxis(sr)
        .add_yaxis("← 新用户",nv,color="#24477f",label_opts=lbl_l,gap="-100%")
        .add_yaxis("老用户 →",ov,color="#5a8bc9",label_opts=lbl_r,gap="-100%")
        .reversal_axis().set_global_opts(
            title_opts=opts.TitleOpts(title=title,title_textstyle_opts=opts.TextStyleOpts(font_size=13)),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(font_size=10)),
            xaxis_opts=opts.AxisOpts(min_=-max_val,max_=max_val,
                axislabel_opts=opts.LabelOpts(formatter=JsCode("function(v){return Math.abs(v)+'%';}"))),
            legend_opts=opts.LegendOpts(pos_top="30px",textstyle_opts=opts.TextStyleOpts(font_size=10)),
            tooltip_opts=opts.TooltipOpts(trigger="axis",axis_pointer_type="shadow",
                formatter=JsCode(tooltip_fn))))
    grid = Grid(init_opts=opts.InitOpts(width="520px",height=f"{h}px")).add(
        bar, grid_opts=opts.GridOpts(pos_left="28%", pos_right="8%"))
    return _ch(grid)

# ─── 策略气泡图 ──────────────────────────────────────────────

def chart_s7_bubble(gd):
    df = gd.copy()
    med_x = float(df['占比(%)'].median())
    med_y = float(df['下载点击率(%)'].median())
    med_z = float(df['下载成功率(%)'].median())
    def strat(r):
        big=r['占比(%)']>=med_x; good=r['下载点击率(%)']>=med_y; pay=r['下载成功率(%)']>=med_z
        if big and good and pay: return '核心现金牛'
        if big and good and not pay: return '商业化优化'
        if big and not good: return '算法优化机会'
        if not big and good and pay: return '小众高价值'
        if not big and good and not pay: return '小众优质'
        return '低优先级'
    df['strategy'] = df.apply(strat, axis=1)
    colors = {'核心现金牛':'#4cb573','商业化优化':'#6b9bd4','算法优化机会':'#e87888',
              '小众高价值':'#d49a4a','小众优质':'#9a8aba','低优先级':'#8a9aac'}
    share_max = max(float(df['占比(%)'].max()), 0.1)
    # 仅对占比突出的气泡显示标签：占比 >= 全表 50% 分位
    share_label_threshold = float(df['占比(%)'].quantile(0.5))
    scatter = Scatter(init_opts=opts.InitOpts(width="1020px",height="560px"))
    scatter.add_xaxis(xaxis_data=[])
    tooltip_js = JsCode("function(p){var d=p.data;return '<b>'+d[4]+'</b><br/>下载点击率:'+d[0]+'%<br/>付费转化率:'+d[1]+'%<br/>场景占比:'+d[2]+'%<br/>需求量:'+d[3]+'人';}")
    for st in ['核心现金牛','商业化优化','算法优化机会','小众高价值','小众优质','低优先级']:
        sub = df[df['strategy']==st]
        if sub.empty: continue
        # 每行: [点击率, 成功率, 占比, count, scene_l2, 是否显示标签 1/0]
        data = [[round(float(r['下载点击率(%)']),1), round(float(r['下载成功率(%)']),1),
                 round(float(r['占比(%)']),2), int(r['count']), r['scene_l2'],
                 1 if float(r['占比(%)']) >= share_label_threshold else 0] for _,r in sub.iterrows()]
        # 下载点击率 >= med_y 标签放右侧，< med_y 放左侧
        data_left = [row for row in data if row[0] < med_y]
        data_right = [row for row in data if row[0] >= med_y]
        if data_left:
            scatter.add_yaxis(st, data_left,
                symbol_size=JsCode("function(val){return Math.max(8, val[2]/"+f"{share_max:.1f}"+"*45+6);}"),
                itemstyle_opts=opts.ItemStyleOpts(color=colors[st],opacity=.82),
                label_opts=opts.LabelOpts(is_show=True, position="left", font_size=11, color="#1f2937", distance=4,
                    formatter=JsCode("function(p){var d=p.data;return (d&&d[5])? (d[4]||''):'';}")),
                tooltip_opts=opts.TooltipOpts(formatter=tooltip_js))
        if data_right:
            scatter.add_yaxis(st, data_right,
                symbol_size=JsCode("function(val){return Math.max(8, val[2]/"+f"{share_max:.1f}"+"*45+6);}"),
                itemstyle_opts=opts.ItemStyleOpts(color=colors[st],opacity=.82),
                label_opts=opts.LabelOpts(is_show=True, position="right", font_size=11, color="#1f2937", distance=4,
                    formatter=JsCode("function(p){var d=p.data;return (d&&d[5])? (d[4]||''):'';}")),
                tooltip_opts=opts.TooltipOpts(formatter=tooltip_js))
    scatter.set_series_opts(
        markline_opts=opts.MarkLineOpts(
            data=[
                opts.MarkLineItem(x=med_y, name=f'点击率中位数 {med_y:.0f}%'),
                opts.MarkLineItem(y=med_z, name=f'付费转化率中位数 {med_z:.0f}%'),
            ],
            linestyle_opts=opts.LineStyleOpts(type_='dashed', color='#24477f', width=3),
            label_opts=opts.LabelOpts(is_show=True, position='end', color='#24477f', font_weight='bold', font_size=12),
        ),
        label_layout={"hideOverlap": True},
    )
    scatter.set_global_opts(
        title_opts=opts.TitleOpts(title="场景策略地图"),
        xaxis_opts=opts.AxisOpts(name="下载点击率(%)",type_="value",min_=0,max_=100,
            axislabel_opts=opts.LabelOpts(formatter="{value}%"),
            splitline_opts=opts.SplitLineOpts(is_show=True,linestyle_opts=opts.LineStyleOpts(opacity=.12))),
        yaxis_opts=opts.AxisOpts(name="付费转化率(%)",type_="value",min_=0,max_=100,
            axislabel_opts=opts.LabelOpts(formatter="{value}%"),
            splitline_opts=opts.SplitLineOpts(is_show=True,linestyle_opts=opts.LineStyleOpts(opacity=.12))),
        legend_opts=opts.LegendOpts(pos_top="32px",pos_right="2%",orient="vertical",item_width=14,item_height=12,
            textstyle_opts=opts.TextStyleOpts(font_size=11)))
    raw = _ch(scatter)
    subtitle_html = (f'<p style="font-size:13px;color:#6b7280;margin-top:6px;text-align:center">'
                     f'横轴=下载点击率 &nbsp; 纵轴=付费转化率 &nbsp; 气泡大小=场景占比 &nbsp;|&nbsp; '
                     f'虚线=中位数分界（点击率 {med_y:.0f}% / 付费转化率 {med_z:.0f}%）</p>')
    return raw + subtitle_html, med_x, med_y, med_z

# ─── 新老用户场景差异表（来自 新老用户.md） ─────────────────────

def build_new_old_user_table(md_path='statistics/新老用户.md'):
    """解析 新老用户.md 的表格（分类、场景、新用户占比、老用户占比、核心判断），输出 HTML 表。"""
    if not os.path.isfile(md_path):
        return ''
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    except Exception:
        return ''
    if len(lines) < 2:
        return ''
    # 表头
    header = lines[0].split('\t')
    if len(header) < 5:
        return ''
    rows = []
    for ln in lines[1:]:
        cells = ln.split('\t')
        cells = (cells + [''] * 5)[:5]
        rows.append(cells)
    if not rows:
        return ''
    # 分类列：空表示延续上一类，非空为新类
    groups = []  # [(分类名, start_idx, count)]
    for i, r in enumerate(rows):
        cat = (r[0].strip() if r else '')
        if cat:
            groups.append((cat, i, 1))
        else:
            if groups:
                groups[-1] = (groups[-1][0], groups[-1][1], groups[-1][2] + 1)
    # 若第一行分类为空，则整表为一组
    if not groups and rows:
        groups = [('', 0, len(rows))]
    tbl = []
    tbl.append('<div class="tbl-wrap" style="margin-top:16px"><table><thead><tr>')
    tbl.append('<th>分类</th><th>场景</th><th>新用户占比</th><th>老用户占比</th><th>核心判断</th>')
    tbl.append('</tr></thead><tbody>')
    for i, r in enumerate(rows):
        cat, scene, pct_new, pct_old, judge = (r[0].strip(), r[1].strip(), r[2].strip(), r[3].strip(), r[4].strip())
        # 找当前行所属分组及是否为该组首行
        group_rowspan = None
        show_cat = None
        for g_name, g_start, g_count in groups:
            if g_start <= i < g_start + g_count:
                if i == g_start:
                    group_rowspan = g_count
                    show_cat = g_name
                break
        cat_td = f'<td rowspan="{group_rowspan}" style="font-weight:600;text-align:center;vertical-align:middle">{show_cat}</td>' if show_cat is not None else ''
        tbl.append(f'<tr>{cat_td}<td>{scene}</td><td>{pct_new}</td><td>{pct_old}</td><td>{judge}</td></tr>')
    tbl.append('</tbody></table></div>')
    return '\n'.join(tbl)

# ─── 附录：抠图打标分类体系（用户可读） ─────────────────────────

def build_classification_appendix(json_path='图像识别_副本/分类体系/抠图打标分类体系_v4.json', extra_after_part1=None):
    """从 v4 分类体系 JSON 生成附录 HTML。extra_after_part1 若提供则插入为附录第二部分（如主体分析）。"""
    if not os.path.isfile(json_path):
        return '<p class="muted">附录：分类体系文件未找到（%s）</p>' % json_path
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return '<p class="muted">附录：分类体系解析失败（%s）</p>' % str(e)
    out = []
    # 附录一大块：一、抠图打标分类体系（其下三块用 h4）
    out.append('<div class="card" style="margin-top:16px"><h3>一、抠图打标分类体系</h3>')
    out.append('<h4 style="margin:14px 0 8px;font-size:15px;color:var(--blue)">场景分类（用途）</h4>')
    out.append('<p style="color:var(--muted);font-size:13px">按图片<strong>使用场景</strong>划分：用户拿抠图去做什么。</p>')
    scenes = data.get('scene_classification') or {}
    for l1, info in scenes.items():
        desc = (info.get('description') or '').strip()
        subs = info.get('subcategories') or []
        out.append(f'<div style="margin-bottom:14px"><strong style="color:var(--blue)">{l1}</strong>')
        if desc:
            out.append(f'<p style="margin:4px 0 6px;font-size:13px;color:#334155">{desc}</p>')
        out.append('<p style="margin:0;font-size:13px"><span style="color:var(--muted)">子类：</span>' + '、'.join(subs) + '</p></div>')
    out.append('<h4 style="margin:18px 0 8px;font-size:15px;color:var(--green)">主体分类（抠什么）</h4>')
    out.append('<p style="color:var(--muted);font-size:13px">按图片<strong>主体</strong>划分：被抠的对象是什么。</p>')
    subjects = data.get('subject_classification') or {}
    for l1, info in subjects.items():
        desc = (info.get('description') or '').strip()
        subs = info.get('subcategories') or []
        out.append(f'<div style="margin-bottom:14px"><strong style="color:var(--green)">{l1}</strong>')
        if desc:
            out.append(f'<p style="margin:4px 0 6px;font-size:13px;color:#334155">{desc}</p>')
        out.append('<p style="margin:0;font-size:13px"><span style="color:var(--muted)">子类：</span>' + '、'.join(subs) + '</p></div>')
    out.append('<h4 style="margin:18px 0 8px;font-size:15px;color:var(--sub)">复杂度标签</h4>')
    out.append('<p style="color:var(--muted);font-size:13px">描述抠图<strong>技术难度</strong>，可多选。</p>')
    tags = data.get('difficulty_tags') or []
    out.append('<p style="margin:0;font-size:13px">' + '、'.join(tags) + '</p></div>')
    if extra_after_part1:
        out.append(extra_after_part1)
    return '\n'.join(out)

# ─── 4. 策略矩阵 ────────────────────────────────────────────

def strategy_table(gd, med_x, med_y, med_z, scene_to_image=None, url_to_local=None):
    df = gd.copy()
    scene_to_image = scene_to_image or {}
    url_to_local = url_to_local or {}
    strat_map = {
        '核心现金牛':  ('#dcfce7','#166534','需求大+下载多+付费多，重点强化'),
        '商业化优化':  ('#dbeafe','#1e40af','需求大+下载多+付费少，激活付费'),
        '算法优化机会': ('#fee2e2','#991b1b','需求大+下载多，算法/体验优化'),
        '小众高价值':  ('#fef3c7','#92400e','需求小+下载多+付费多，溢价/拓展'),
        '小众优质':    ('#ede9fe','#5b21b6','需求小+下载多+付费少，维持/尝试商业化'),
        '低优先级':    ('#f1f5f9','#475569','需求小+下载少，暂不投入'),
    }
    def st(r):
        big=r['占比(%)']>=med_x; good=r['下载点击率(%)']>=med_y; pay=r['下载成功率(%)']>=med_z
        if big and good and pay: return '核心现金牛'
        if big and good and not pay: return '商业化优化'
        if big and not good: return '算法优化机会'
        if not big and good and pay: return '小众高价值'
        if not big and good and not pay: return '小众优质'
        return '低优先级'
    df['_st']=df.apply(st,axis=1)
    rows = ''
    for strat_name in ['核心现金牛','商业化优化','算法优化机会','小众高价值','小众优质','低优先级']:
        sub = df[df['_st']==strat_name].sort_values('count',ascending=False)
        if sub.empty: continue
        bg,fc,tip = strat_map[strat_name]
        for i,(_,row) in enumerate(sub.iterrows()):
            sc = (f'<td rowspan="{len(sub)}" style="background:{bg};color:{fc};font-weight:700;text-align:center">'
                  f'{strat_name}<br><span style="font-size:11px;font-weight:400">{tip}</span></td>' if i==0 else '')
            scene_key = str(row['scene_l2']).strip() if pd.notna(row.get('scene_l2')) else ''
            url = scene_to_image.get(scene_key, '') if scene_key else ''
            url = url_to_local.get(url, url) if url else ''
            if not isinstance(url, str):
                url = ''
            url_esc = url.replace('"', '&quot;').replace("'", '&#39;') if url else ''
            thumb = f'<span class="thumb-zoom" data-src="{url_esc}" onclick="window.thumbModalShow(this)"><img class="report-thumb" src="{url}" alt="" onerror="this.style.display=\'none\'" /></span>' if url else '—'
            rows += (f'<tr>{sc}<td>{thumb}</td><td>{row["scene_l2"]}</td><td>{int(row["count"])}</td>'
                     f'<td>{row["占比(%)"]:.2f}%</td><td>{row["下载点击率(%)"]:.1f}%</td>'
                     f'<td style="color:#7c3aed;font-weight:600">{row["下载成功率(%)"]:.1f}%</td></tr>')
    return f'''<div class="tbl-wrap"><table><thead><tr><th>策略分区</th><th>缩略图</th><th>Scene L2</th><th>需求量</th>
<th>占比</th><th>下载点击率</th><th>下载成功率(付费率)</th></tr></thead><tbody>{rows}</tbody></table></div>'''

# ─── 5. CSS ──────────────────────────────────────────────────

CSS = """<style>
/* McKinsey Consulting 风格 — 深蓝结构化说服力，内容不改 */
:root{--blue:#24477f;--blue-l:#e8eef5;--green:#0d6b2c;--green-l:#f0fdf4;
  --red:#c41e3a;--red-l:#fef2f2;--orange:#b85c00;--orange-l:#fff7ed;
  --gray:#4a5568;--gray-l:#f5f6f8;--gray-b:#e0e2e5;--text:#1a1a1a;
  --sub:#4a5568;--muted:#5a6578;--shadow:0 1px 0 rgba(0,0,0,.06)}
*{box-sizing:border-box;margin:0;padding:0}
html{background:#f5f6f8}
body{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;background:#f5f6f8;color:var(--text);font-size:16px;line-height:1.8;max-width:1200px;margin:0 auto;padding:40px 48px}
.hero-top-line{height:4px;background:#24477f;width:100%}
/* ─ 侧边栏目录 ─ */
.toc{position:fixed;left:0;top:0;height:100vh;width:240px;background:#fff;box-shadow:2px 0 12px rgba(36,71,127,.12);border-right:1px solid #e0e2e5;
  z-index:1000;transition:transform .3s;overflow-y:auto;padding:16px 0 20px}
.toc.collapsed{transform:translateX(-240px)}
.toc-head{display:flex;justify-content:space-between;align-items:center;padding:10px 16px;border-bottom:1px solid #e0e2e5;margin-bottom:6px}
.toc-head span{font-weight:700;font-size:15px;color:#1a1a1a}
.toc-close{background:none;border:none;cursor:pointer;font-size:18px;color:#5a6578;padding:2px 6px}
.toc-close:hover{color:#24477f}
.toc a{display:block;padding:7px 16px;font-size:13px;color:#1a1a1a;text-decoration:none;border-left:3px solid transparent;transition:all .15s}
.toc a:hover,.toc a.active{background:#e8eef5;border-left-color:#24477f;color:#24477f;font-weight:600}
.toc-open{position:fixed;left:8px;top:8px;z-index:999;background:#24477f;color:#fff;border:none;border-radius:4px;
  padding:8px 14px;cursor:pointer;font-size:13px;font-weight:600;box-shadow:none;
  transition:opacity .3s}
.toc-open:hover{background:#1a3560}
/* ─ 主体 ─ */
.hero{background:#fff;color:#1a1a1a;padding:32px 0 24px;border-bottom:1px solid #e0e2e5;margin:0 -48px 24px -48px;padding-left:48px;padding-right:48px}
.hero h1{font-size:26px;font-weight:700;font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;color:#1a1a1a}
.hero .sub{opacity:.9;font-size:15px;margin-top:8px;color:var(--sub)}
.hero .meta{display:flex;gap:14px;flex-wrap:wrap;margin-top:16px}
.hero .meta span{background:#e8eef5;border:1px solid #e0e2e5;border-radius:4px;padding:6px 14px;font-size:13px;font-weight:500;color:#1a1a1a}
.container{max-width:1200px;margin:0 auto;padding:0}
.section{margin-bottom:48px;scroll-margin-top:20px}
h2{font-size:20px;font-weight:700;font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;border-left:4px solid #24477f;padding-left:12px;margin-bottom:16px;color:#1a1a1a}
h3{font-size:16px;font-weight:700;color:var(--text);margin:12px 0 8px}
p,li{color:var(--sub);font-size:15px;line-height:1.85}
strong{color:var(--text)}
.card{background:#fff;border:1px solid #e0e2e5;border-radius:4px;box-shadow:none;padding:24px 26px;margin-bottom:16px}
.insight{border-radius:4px;padding:14px 18px;margin-bottom:12px;border:1px solid #e0e2e5}
.insight.blue{background:#fffde7;border-left:4px solid #24477f}
.insight.purple{background:#fff9c4;border-left:4px solid #5a4a7a}
.insight.plain{background:#fff;border:1px solid #e0e2e5;border-left:4px solid #5a4a7a}
.insight .ttl{font-weight:700;font-size:15px;margin-bottom:5px;color:var(--text)}
.insight p{margin:0;font-size:14px;line-height:1.75}
.tbl-wrap{overflow-x:auto;border-radius:4px;margin-top:10px;border:1px solid #e0e2e5}
.tbl-wrap--sticky-head{max-height:750px;overflow-y:auto;overflow-x:auto}
.tbl-wrap--sticky-head thead th{position:sticky;top:0;z-index:2;background:#24477f;box-shadow:0 1px 0 #e0e2e5}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#24477f;color:#fff;font-weight:700;padding:10px 12px;text-align:left;border-bottom:1px solid #e0e2e5;white-space:nowrap}
td{padding:9px 12px;border-bottom:1px solid #e0e2e5}
tr:nth-child(even) td{background:#f8f9fb}
tr:hover td{background:#e8eef5}
tr.row-ratio-gt1 td{background:#fff59d !important;font-weight:600}
.chart-wrap{background:#fff;border:1px solid #e0e2e5;border-radius:4px;padding:12px 8px;margin-bottom:16px;overflow-x:auto}
.chart-scroll-x{overflow-x:auto;-webkit-overflow-scrolling:touch;margin-bottom:16px;padding-bottom:8px}
.chart-scroll-x>.chart-wrap{min-width:1060px}
.chart-scroll-x>.grid-2{min-width:1052px}
.cross-analysis-charts .chart-wrap{overflow-x:auto;min-width:0}
.cross-analysis-charts .grid-2{min-width:0}
.cross-analysis-charts.chart-scroll-x{overflow-x:auto}
.cross-analysis-charts.chart-scroll-x .grid-2{min-width:1052px}
.cross-analysis-charts.chart-scroll-x .chart-wrap{min-width:520px}
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:16px}
.hero .meta#sec-kpi{margin-top:12px}
.kpi{background:#fff;border:1px solid #e0e2e5;border-radius:4px;padding:20px;text-align:center}
.kpi .v{font-size:32px;font-weight:800;font-variant-numeric:tabular-nums}.kpi .l{font-size:13px;color:var(--muted);margin-top:4px;font-weight:500}
.kpi .h{font-size:12px;color:var(--muted)}.kpi.b .v{color:#24477f}.kpi.g .v{color:#0d6b2c}
.kpi.o .v{color:#b85c00}.kpi.p .v{color:#5a4a7a}
.footer{text-align:center;color:var(--muted);font-size:12px;padding:24px 0}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.conclusion{background:#fffde7;border-left:4px solid #24477f;border-radius:4px;padding:14px 18px;margin:12px 0;border:1px solid #f5e082;border-left-width:4px}
.conclusion .ttl{font-weight:700;font-size:14px;color:#24477f;margin-bottom:4px}
.conclusion p{font-size:14px;color:#1a1a1a;margin:0;line-height:1.75}
.exec-summary{background:#fffde7;border:1px solid #f5e082;border-radius:4px;padding:24px 28px;margin-bottom:24px;border-top:4px solid #24477f;scroll-margin-top:20px}
.exec-summary.no-highlight{background:#fff;border:1px solid #e0e2e5}
.exec-summary.no-highlight .exec-item{background:#f8f9fb;border-color:#e0e2e5}
.exec-summary h3{font-size:18px;font-weight:700;font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;color:#1a1a1a;margin-bottom:12px}
.exec-summary .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.exec-item{border-radius:4px;padding:12px 16px;border:1px solid #f5e082;background:#fff9c4}
.exec-item .tag{font-size:12px;font-weight:700;margin-bottom:4px;text-transform:uppercase}
.exec-item .item-ttl{font-weight:700;color:#24477f}
.exec-item .sub-ttl{font-weight:600;font-size:12px;color:var(--sub)}
.exec-item p{font-size:13px;margin:0;line-height:1.7}
details{margin-top:8px}
details summary{cursor:pointer;font-weight:600;font-size:14px;color:#24477f;padding:8px 0;user-select:none}
details summary:hover{color:#1a3560}
.report-thumb{width:56px;height:56px;object-fit:cover;border-radius:4px;display:block;background:#e8eef5;cursor:pointer}
.thumb-zoom{cursor:pointer;display:inline-block}
.grid-2.wc-equal-height{align-items:stretch}
.grid-2.wc-equal-height>.wc-with-examples{display:flex;flex-direction:column;min-height:0}
.thumb-modal{display:none;position:fixed;left:0;top:0;width:100%;height:100%;background:rgba(0,0,0,.75);z-index:2000;justify-content:center;align-items:center;cursor:pointer}
.thumb-modal.show{display:flex}
.thumb-modal img{max-width:95%;max-height:95%;object-fit:contain;border-radius:8px;pointer-events:none}
@media(max-width:700px){.kpi-row,.grid-2,.exec-summary .grid{grid-template-columns:1fr}}
</style>"""

ECHARTS_JS = '<script src="https://assets.pyecharts.org/assets/v5/echarts.min.js"></script>'

TOC_JS = """<script>
document.querySelectorAll('.toc a').forEach(function(a){
  a.addEventListener('click',function(e){
    e.preventDefault();
    var t=document.querySelector(this.getAttribute('href'));
    if(t) t.scrollIntoView({behavior:'smooth',block:'start'});
    if(window.innerWidth<900) document.getElementById('toc').classList.add('collapsed');
  });
});
window.addEventListener('scroll',function(){
  var secs=document.querySelectorAll('[id^="sec-"]');
  var links=document.querySelectorAll('.toc a');
  var cur='';
  secs.forEach(function(s){if(s.getBoundingClientRect().top<=120) cur='#'+s.id;});
  links.forEach(function(a){a.classList.toggle('active',a.getAttribute('href')===cur);});
});
function tocToggle(show){
  var el=document.getElementById('toc');
  if(show) el.classList.remove('collapsed'); else el.classList.add('collapsed');
}
window.thumbModalShow=function(el){
  var src=el.dataset.src||(el.querySelector&&el.querySelector('img')&&el.querySelector('img').src);
  if(!src) return;
  var m=document.getElementById('thumb-modal');
  if(!m){ m=document.createElement('div'); m.id='thumb-modal'; m.className='thumb-modal'; m.innerHTML='<img src="" alt="">'; m.onclick=function(){ m.classList.remove('show'); }; document.body.appendChild(m); }
  m.querySelector('img').src=src; m.classList.add('show');
};
</script>"""

# ─── 6. 主流程 ───────────────────────────────────────────────

def main():
    print("加载数据...")
    df = load_data(); g = agg_l2(df)
    gd = display_filter(g)
    N=len(df); tc=int(df['is_download_click'].sum()); ts=int(df['is_saved'].sum()); tp=int(df['paid_for_url'].sum())
    tc_safe = tc if tc else 1  # 新增付费率分母（下载点击数）
    # 下载成功率平均值：各场景下载成功率（该场景 下载成功数/下载点击数）再求平均，非 sum(成功)/sum(点击)
    g_with_clicks = g[g['click_count'] > 0]
    avg_save_rate = float(g_with_clicks['下载成功率(%)'].mean()) if len(g_with_clicks) else 0.0
    top3=g.head(3)['scene_l2'].tolist(); top10_share=g.head(10)['占比(%)'].sum()
    n_new=(df['user_type']=='新用户').sum(); n_old=(df['user_type']=='老用户').sum()

    print("生成图表...")
    c1=chart_s1(g)
    c2=chart_s2(gd); c3=chart_s3(gd); c4=chart_s4(gd)
    c5a=chart_s5_subject(df)

    # ── 策略分类（需在场景选择前计算） ──
    g_no_undet = g[g['scene_l2']!='未确定']
    med_x = float(gd['占比(%)'].median())
    med_y = float(gd['下载点击率(%)'].median())
    med_z = float(gd['下载成功率(%)'].median())
    def _st_label(r):
        big=r['占比(%)']>=med_x; good=r['下载点击率(%)']>=med_y; pay=r['下载成功率(%)']>=med_z
        if big and good and pay: return '核心现金牛'
        if big and good and not pay: return '商业化优化'
        if big and not good: return '算法优化机会'
        if not big and good and pay: return '小众高价值'
        if not big and good and not pay: return '小众优质'
        return '低优先级'
    g15=gd.copy(); g15['策略']=g15.apply(_st_label,axis=1)
    
    # 使用策略分类选择场景
    core_scenes = g15[g15['策略']=='核心现金牛']['scene_l2'].tolist()
    biz_opt_scenes = g15[g15['策略']=='商业化优化']['scene_l2'].tolist()
    algo_scenes = g15[g15['策略']=='算法优化机会']['scene_l2'].tolist()
    # 为每类场景分别添加小标题（每类只加一次标题）
    c5b = []
    # 核心现金牛场景
    core_has_chart = False
    for sc in core_scenes:
        chart = chart_s5_sub(df, sc)
        if chart:
            if not core_has_chart:
                c5b.append(f'<h4 style="font-weight:700;color:#0d6b2c;margin:8px 0 4px;font-size:14px">核心现金牛场景</h4>')
                core_has_chart = True
            c5b.append(chart)
    # 商业化优化场景
    biz_has_chart = False
    for sc in biz_opt_scenes:
        chart = chart_s5_sub(df, sc)
        if chart:
            if not biz_has_chart:
                c5b.append(f'<h4 style="font-weight:700;color:#24477f;margin:8px 0 4px;font-size:14px">商业化优化场景</h4>')
                biz_has_chart = True
            c5b.append(chart)
    # 算法优化机会场景
    algo_has_chart = False
    for sc in algo_scenes:
        chart = chart_s5_sub(df, sc)
        if chart:
            if not algo_has_chart:
                c5b.append(f'<h4 style="font-weight:700;color:#c41e3a;margin:8px 0 4px;font-size:14px">算法优化机会场景</h4>')
                algo_has_chart = True
            c5b.append(chart)

    # ── 主体分布分析 ──
    def scene_subject_info(scene):
        sub = df[df['scene_l2']==scene].dropna(subset=['subject_l2'])
        gs = sub.groupby('subject_l2').size().reset_index(name='cnt').sort_values('cnt',ascending=False)
        total = gs['cnt'].sum()
        gs['share'] = (gs['cnt']/total*100).round(1)
        top1_share = float(gs.iloc[0]['share']) if len(gs)>0 else 0
        top2_share = float(gs.head(2)['share'].sum()) if len(gs)>=2 else top1_share
        top3_share = float(gs.head(3)['share'].sum()) if len(gs)>=3 else top2_share
        top_subs = gs.head(3)['subject_l2'].tolist()
        n_subs = len(gs)
        if top1_share > 50:
            dtype = f"高度集中，{top1_share:.0f}% 为「{gs.iloc[0]['subject_l2']}」"
        elif top3_share > 60:
            dtype = f"较集中，TOP3 主体占 {top3_share:.0f}%"
        else:
            dtype = f"相对分散，{n_subs} 个主体"
        return top_subs, dtype, top1_share, top3_share

    core_info = {s: scene_subject_info(s) for s in core_scenes}
    biz_opt_info = {s: scene_subject_info(s) for s in biz_opt_scenes}
    algo_info = {s: scene_subject_info(s) for s in algo_scenes}
    # 执行摘要用：算法优化机会场景及主要优化主体（指定场景填入给定主体，其余用数据）
    algo_optimize_subjects = {
        "商品详情页": "家具",
        "电商主图": "上装",
        "电商海报/活动页": "医疗用品、家用电器",
        "文档配图": "公章印章、书籍/教材/试卷/流程图",
        "海报设计":"待观察"
    }
    def _algo_p0_label(sc):
        sub = algo_optimize_subjects.get(sc)
        if sub is not None:
            return f"{sc}（主要优化主体：{sub}）"
        if algo_info[sc][0]:
            return f"{sc}（主要优化主体：{'、'.join(algo_info[sc][0][:1])}）"
        return sc
    algo_p0_parts = [_algo_p0_label(sc) for sc in algo_scenes]
    algo_p0_text = '；'.join(algo_p0_parts) if algo_p0_parts else '无'

    # ── difficulty_tags ──
    # 高保存场景 = 核心现金牛中的场景
    # 低保存场景 = 算法优化机会中的场景
    # 高保存场景包括核心现金牛和商业化优化场景
    high_save_scenes = core_scenes + biz_opt_scenes
    core_tags_counter = _parse_tags(df[df['scene_l2'].isin(high_save_scenes)]['difficulty_tags'])
    algo_tags_counter = _parse_tags(df[df['scene_l2'].isin(algo_scenes)]['difficulty_tags'])
    wc_core = _html_wordcloud(core_tags_counter, f"高保存场景（核心现金牛+商业化优化）（{'、'.join(high_save_scenes[:4])}）")
    wc_algo = _html_wordcloud(algo_tags_counter, f"低保存场景（算法优化机会）（{'、'.join(algo_scenes[:4])}）")
    all_tags_ordered = [t for t, _ in core_tags_counter.most_common(15)] + [t for t, _ in algo_tags_counter.most_common(15) if t not in core_tags_counter]
    used_urls = set()
    tag_to_image = {}
    for tag in all_tags_ordered:
        if tag in tag_to_image:
            continue
        sub = df[df['difficulty_tags'].apply(lambda x: tag in _parse_row_tags(x))]
        if not sub.empty:
            # 简单边缘：优先选仅含该标签或标签最少的行，使示例图更典型
            if tag == '简单边缘':
                rows_with_tag = [(_, r) for _, r in sub.iterrows()]
                def _tag_count(row):
                    tgs = _parse_row_tags(row['difficulty_tags'])
                    return len(tgs), (1 if tgs == ['简单边缘'] else 0)
                rows_with_tag.sort(key=lambda x: (-_tag_count(x[1])[1], _tag_count(x[1])[0]))
            else:
                rows_with_tag = [(_, r) for _, r in sub.iterrows()]
            for _, r in rows_with_tag:
                u = r['image_url']
                u = _norm_url(u)
                if u and u not in used_urls:
                    used_urls.add(u)
                    tag_to_image[tag] = u
                    break
    def _tag_examples_html(tag_to_image, url_to_local=None):
        if not tag_to_image:
            return ''
        url_to_local = url_to_local or {}
        parts = []
        for tag, url in tag_to_image.items():
            url = url_to_local.get(url, url) if url else ''
            if not isinstance(url, str):
                url = ''
            url_esc = url.replace('"', '&quot;').replace("'", '&#39;') if url else ''
            thumb = f'<span class="thumb-zoom" data-src="{url_esc}" onclick="window.thumbModalShow(this)"><img class="report-thumb" src="{url}" alt="" onerror="this.style.display=\'none\'" /></span>' if url else ''
            parts.append(f'<div style="text-align:center"><span style="font-size:12px;color:var(--sub)">{tag}</span><br>{thumb}</div>')
        return '<div style="font-size:14px;font-weight:700;color:var(--sub);margin-bottom:10px">图片举例</div><div style="display:flex;flex-wrap:wrap;gap:10px;justify-content:center">' + ''.join(parts) + '</div>'
    wc_core_block = ('<div class="wc-with-examples">' + wc_core + '</div>') if wc_core else ''
    wc_algo_block = ('<div class="wc-with-examples">' + wc_algo + '</div>') if wc_algo else ''
    wc_tag_examples_card = None  # 在 url_to_local 构建后赋值
    complex_tags = {'复杂边缘','发丝/毛发','透明/半透明','前背景色接近','反光/高光'}
    core_total = sum(core_tags_counter.values()) or 1; algo_total = sum(algo_tags_counter.values()) or 1
    core_complex_pct = sum(core_tags_counter.get(t,0) for t in complex_tags)/core_total*100
    algo_complex_pct = sum(algo_tags_counter.get(t,0) for t in complex_tags)/algo_total*100
    core_simple_pct = core_tags_counter.get('简单边缘',0)/core_total*100
    algo_simple_pct = algo_tags_counter.get('简单边缘',0)/algo_total*100
    core_top5 = [t for t,_ in core_tags_counter.most_common(5)]
    algo_top5 = [t for t,_ in algo_tags_counter.most_common(5)]

    cross_lines = []
    cross_lines.append("<b>核心现金牛场景主体分布：</b>")
    for s in core_scenes:
        subs, dtype, t1_share, t3_share = core_info[s]
        if t3_share < 50:
            cross_lines.append(f"· 「{s}」{dtype}，TOP3主体占比{t3_share:.0f}%，主要主体：{'、'.join(subs[:3])}")
        else:
            cross_lines.append(f"· 「{s}」{dtype}，TOP3主体占比{t3_share:.0f}%。主要主体：{'、'.join(subs[:3])}")
    cross_lines.append("<b>商业化优化场景主体分布：</b>")
    for s in biz_opt_scenes:
        subs, dtype, t1_share, t3_share = biz_opt_info[s]
        if t3_share < 50:
            cross_lines.append(f"· 「{s}」{dtype}，TOP3主体占比{t3_share:.0f}%，主要主体：{'、'.join(subs[:3])}")
        else:
            cross_lines.append(f"· 「{s}」{dtype}，TOP3主体占比{t3_share:.0f}%。主要主体：{'、'.join(subs[:3])}")
    cross_lines.append("<b>算法优化机会场景主体分布：</b>")
    for s in algo_scenes:
        subs, dtype, t1_share, t3_share = algo_info[s]
        if t3_share < 50:
            cross_lines.append(f"· 「{s}」{dtype}，TOP3主体占比{t3_share:.0f}%，主要主体：{'、'.join(subs[:3])}")
        else:
            cross_lines.append(f"· 「{s}」{dtype}，TOP3主体占比{t3_share:.0f}%。主要主体：{'、'.join(subs[:3])}")
    cross_lines.append("<b>复杂度标签 对比：</b>")
    cross_lines.append(f"· 高需求高保存场景（核心现金牛+商业化优化）TOP3：{'、'.join(core_top5[:3])}，简单边缘占 {core_simple_pct:.0f}%，复杂标签合计 {core_complex_pct:.0f}%")
    cross_lines.append(f"· 高需求低保存场景（算法优化机会）TOP3：{'、'.join(algo_top5[:3])}，简单边缘占 {algo_simple_pct:.0f}%，复杂标签合计 {algo_complex_pct:.0f}%")
    if algo_complex_pct > core_complex_pct + 5:
        cross_lines.append(f"· <b>算法优化机会场景复杂标签占比高出 {algo_complex_pct-core_complex_pct:.0f}pp</b>，<b>技术难度是核心瓶颈</b>。")
    elif core_complex_pct > algo_complex_pct + 5:
        cross_lines.append(f"· 高需求高保存场景复杂标签反而更多（{core_complex_pct:.0f}% vs {algo_complex_pct:.0f}%），低保存可能源于主体类型或用户预期差异。")
    else:
        cross_lines.append(f"· 两类场景复杂标签占比接近，可能需要从主体类型或用户行为角度分析差异。")

    # 场景×主体 需求占比与保存/点击，用于结论中「可优先优化」归纳
    df_ss_agg = df.dropna(subset=['scene_l2', 'subject_l2']).groupby(['scene_l2', 'subject_l2']).agg(
        count=('device_id', 'count'),
        click_count=('is_download_click', 'sum'),
        saved_count=('is_saved', 'sum'),
    ).reset_index()
    scene_totals = df_ss_agg.groupby('scene_l2')['count'].sum().to_dict()
    df_ss_agg['scene_total'] = df_ss_agg['scene_l2'].map(scene_totals)
    df_ss_agg['需求占比(%)'] = (df_ss_agg['count'] / df_ss_agg['scene_total'] * 100).round(1)
    df_ss_agg['下载点击率(%)'] = (df_ss_agg['click_count'] / df_ss_agg['count'].replace(0, 1) * 100).round(1)
    df_ss_agg['下载成功率(%)'] = (df_ss_agg['saved_count'] / df_ss_agg['click_count'].replace(0, 1) * 100).round(1)

    def _weak_high_save(ss_df, scene_list, max_per_scene=2):
        """高保存场景：需求较高且相对该场景内下载点击率较弱的主体，返回 [(scene, subject), ...]"""
        out = []
        for sc in scene_list:
            s = ss_df[ss_df['scene_l2'] == sc].copy()
            if s.empty or len(s) < 2:
                continue
            med_click = s['下载点击率(%)'].median()
            thresh = s['需求占比(%)'].quantile(0.5)
            high_demand = s[s['需求占比(%)'] >= thresh]
            if high_demand.empty:
                high_demand = s.nlargest(3, '需求占比(%)')
            weak = high_demand[high_demand['下载点击率(%)'] < med_click].nlargest(max_per_scene, '需求占比(%)')
            for _, r in weak.iterrows():
                out.append((r['scene_l2'], r['subject_l2']))
        return out

    def _weak_low_save(ss_df, scene_list, max_per_scene=2):
        """低保存场景：占比较大且下载点击率较低，返回 [(scene, subject), ...]"""
        out = []
        for sc in scene_list:
            s = ss_df[ss_df['scene_l2'] == sc].copy()
            if s.empty or len(s) < 2:
                continue
            med_click = s['下载点击率(%)'].median()
            high_share = s[s['需求占比(%)'] >= 15]
            if high_share.empty:
                high_share = s.nlargest(3, '需求占比(%)')
            weak = high_share[high_share['下载点击率(%)'] < med_click].nlargest(max_per_scene, '需求占比(%)')
            for _, r in weak.iterrows():
                out.append((r['scene_l2'], r['subject_l2']))
        return out

    def _format_weak_by_scene(pairs):
        """将 [(scene, subject), ...] 格式化为「A场景下的a，b主体」；分隔、每场景一行"""
        by_scene = {}
        for sc, sub in pairs:
            by_scene.setdefault(sc, []).append(sub)
        lines = [f"{sc}下的{'，'.join(subs)}主体" for sc, subs in by_scene.items()]
        return "；<br>".join(lines)

    high_save_weak = _weak_high_save(df_ss_agg, high_save_scenes)
    low_save_weak = _weak_low_save(df_ss_agg, algo_scenes)
    high_weak_str = _format_weak_by_scene(high_save_weak) if high_save_weak else ""
    low_weak_str = _format_weak_by_scene(low_save_weak) if low_save_weak else ""

    # TOP3 主体与长尾主体（TOP3 以外）下载点击率，提前计算供结论与明细表使用
    all_table_scenes = core_scenes + biz_opt_scenes + algo_scenes
    top3_click_rate = {}
    long_tail_click_rate = {}
    for sc in all_table_scenes:
        top_subs = scene_subject_info(sc)[0]
        top3_df = df_ss_agg[(df_ss_agg['scene_l2'] == sc) & (df_ss_agg['subject_l2'].isin(top_subs))]
        if top3_df.empty or top3_df['count'].sum() == 0:
            top3_click_rate[sc] = "—"
        else:
            c, cl = top3_df['count'].sum(), top3_df['click_count'].sum()
            rate = (cl / c * 100) if c else 0
            top3_click_rate[sc] = f"{rate:.1f}%"
        tail = df_ss_agg[(df_ss_agg['scene_l2'] == sc) & (~df_ss_agg['subject_l2'].isin(top_subs))]
        if tail.empty or tail['count'].sum() == 0:
            long_tail_click_rate[sc] = "—"
        else:
            c, cl = tail['count'].sum(), tail['click_count'].sum()
            rate = (cl / c * 100) if c else 0
            long_tail_click_rate[sc] = f"{rate:.1f}%"

    # 交叉分析结论：主体分布 + 普适性策略（已删「2、优先优化的主体」）
    cross_conclusion_text = (
        "<strong>1、主体分布</strong><br>"
        "· <b>高保存场景</b>：主体更集中，以各类商品、Logo/品牌标识、卡通插画、生活人像为主。<br>"
        "· <b>低保存场景</b>：主体更分散，多镂空场景，以模特服饰及各类商品、公章印章、文字效果/字体设计/贴纸等为主。<br>"
    )
    # 长尾下载点击率变量保留（供明细表等使用），结论中不再展示长尾段
    algo_with_tail = [sc for sc in algo_scenes if long_tail_click_rate.get(sc) and long_tail_click_rate.get(sc) != "—"]
    scene_rates = gd.set_index('scene_l2')['下载点击率(%)'].to_dict()
    lower_count = sum(1 for sc in algo_with_tail if float(long_tail_click_rate[sc].rstrip('%')) < scene_rates.get(sc, 0))
    # 第三点：普适性策略（字体与上文一致）
    cross_conclusion_text += (
        "<br><strong>2、普适性策略</strong><br>"
        "<span style=\"font-size:1em;font-family:inherit\">"
        "低保存场景的<b>商品图子场景</b>中，<b>长尾</b>（TOP3 以外的主体）保存点击率较低，<b>TOP3 高保存场景</b>保存率较高，"
        "建议对商品主体设置<b>更普适性的策略</b>。</span>"
    )

    # --- 算法优化主体占大盘与投入建议表（场景内保存率口径=下载点击率；按占大盘排序；放在明细表下方、柱状图上方） ---
    df_ss = df.dropna(subset=['subject_l2'])
    total_market = len(df_ss) or 1
    sub_agg = df_ss.groupby('subject_l2').agg(count=('device_id', 'count'), click_count=('is_download_click', 'sum'))
    all_subjects = sub_agg.index.tolist()
    # 阈值：所有二级主体下载点击率的平均值（替代原固定 75%）
    sub_rates = (sub_agg['click_count'] / sub_agg['count'].replace(0, 1) * 100)
    subject_l2_avg_click_rate = round(float(sub_rates.mean()), 1)
    # 场景×主体 聚合，用于计算场景内保存率（下载点击率）
    scene_sub_agg = df.dropna(subset=['scene_l2', 'subject_l2']).groupby(['scene_l2', 'subject_l2']).agg(
        count=('device_id', 'count'), click_count=('is_download_click', 'sum')
    )
    def _scene_rate(sc, full):
        try:
            row = scene_sub_agg.loc[(sc, full)]
            c, cl = int(row['count']), int(row['click_count'])
            return round((cl / (c or 1) * 100), 1)
        except (KeyError, TypeError):
            return None

    def _match_subject_l2(short_name):
        short = (short_name or '').strip().replace('公章印章', '公章/印章')
        for full in all_subjects:
            if full == short or (full.startswith(short) if short else False) or (short in full if short else False):
                return full
        return None

    algo_rows = []
    for sc, sub_str in algo_optimize_subjects.items():
        if (sub_str or '').strip() == '待观察':
            continue
        for short in (sub_str or '').replace('公章印章', '公章/印章').split('、'):
            short = short.strip()
            if not short:
                continue
            full = _match_subject_l2(short)
            scene_rate_val = _scene_rate(sc, full) if full else None
            if full is None:
                algo_rows.append((sc, short + ' [未匹配]', 0.0, 0.0, scene_rate_val, '不优先投入资源优化', False))
                continue
            row = sub_agg.loc[full]
            c, cl = int(row['count']), int(row['click_count'])
            pct_market = (c / total_market * 100)
            click_rate = (cl / (c or 1) * 100)
            if pct_market > 1 and click_rate < subject_l2_avg_click_rate:
                comment = '投入资源整体优化'
                highlight = True
            elif pct_market > 1 and click_rate >= subject_l2_avg_click_rate:
                comment = '重点优化对应场景下的该主体'
                highlight = True
            else:
                comment = '不优先投入资源优化'
                highlight = False
            # 书籍/教材/试卷/流程图 定制文案
            if full == '书籍/教材/试卷/流程图' or (isinstance(full, str) and '书籍/教材/试卷/流程图' in full):
                comment = '文档配图中多为文档+印章形式，和印章同类型一起优化'
            algo_rows.append((sc, full, round(pct_market, 1), round(click_rate, 1), scene_rate_val, comment, highlight))

    # 按占大盘比值从大到小排序
    algo_rows.sort(key=lambda x: x[2], reverse=True)

    algo_table_rows_html = ''
    for sc, sub_full, pct, rate, scene_rate_val, comment, hl in algo_rows:
        tr_class = ' class="row-ratio-gt1"' if (hl or (pct > 1)) else ''
        scene_rate_str = f"{scene_rate_val}%" if scene_rate_val is not None else "—"
        algo_table_rows_html += f'<tr{tr_class}><td>{sc}</td><td>{sub_full}</td><td>{pct}%</td><td>{scene_rate_str}</td><td>{rate}%</td><td>{comment}</td></tr>'
    # 表格 HTML 片段（放入独立卡片，卡片标题为「主体优化方向」）
    algo_subject_table_html = (
        '<div class="tbl-wrap"><table style="width:100%;border-collapse:collapse;font-size:13px">'
        '<thead><tr><th>场景</th><th>对应二级主体</th><th>占大盘比值</th><th>场景内保存率</th><th>大盘下载点击率</th><th>是否投入资源优化</th></tr></thead>'
        f'<tbody>{algo_table_rows_html}</tbody></table></div>'
    )

    # --- 交叉分析明细表（放在结论下方单独展示） ---
    # 第一列：高需求高保存 = 核心现金牛+商业化优化（合并为一大块）；高需求低保存 = 算法优化机会
    save_type_map = {'核心现金牛': '高需求高保存', '商业化优化': '高需求高保存', '算法优化机会': '高需求低保存'}
    strat_groups = {
        '核心现金牛': core_scenes,
        '商业化优化': biz_opt_scenes,
        '算法优化机会': algo_scenes
    }
    strat_colors = {
        '核心现金牛': '#0d6b2c',
        '商业化优化': '#24477f',
        '算法优化机会': '#c41e3a'
    }
    # 每个 scene_l2 取第一个有效 URL；键统一为 str 以便与 gd 行查找一致
    scene_to_image = {}
    for scene, grp in df.groupby('scene_l2'):
        sk = str(scene).strip() if pd.notna(scene) else ''
        if not sk:
            continue
        for _, row in grp.iterrows():
            u = _norm_url(row.get('image_url'))
            if u:
                scene_to_image[sk] = u
                break
    df_ss = df.dropna(subset=['scene_l2', 'subject_l2'])
    scene_subject_to_image = {}
    for (s, sub), grp in df_ss.groupby(['scene_l2', 'subject_l2']):
        sk = (str(s).strip(), str(sub).strip()) if pd.notna(s) and pd.notna(sub) else ('', '')
        if not sk[0] or not sk[1]:
            continue
        for _, row in grp.iterrows():
            u = _norm_url(row.get('image_url'))
            if u:
                scene_subject_to_image[sk] = u
                break
    # 收集报告中用到的全部图片 URL，下载到 statistics/report_thumbs/，供 push 到 GitHub 后展示
    all_thumb_urls = set(scene_to_image.values()) | set(scene_subject_to_image.values()) | set(tag_to_image.values())
    all_thumb_urls.discard('')
    url_to_local = _build_url_to_local(all_thumb_urls)
    if tag_to_image:
        wc_tag_examples_card = ('<div class="card" style="margin-top:12px">' + _tag_examples_html(tag_to_image, url_to_local) + '</div>')
    else:
        wc_tag_examples_card = ''
    def _scene_row_data(sc):
        subs, dtype, t1_share, t3_share = scene_subject_info(sc)
        tags_counter = _parse_tags(df[df['scene_l2'] == sc]['difficulty_tags'])
        top_tags = tags_counter.most_common(3)
        total_tags = sum(tags_counter.values()) or 1
        top_tags_pct_str = '、'.join([f"{t}({c/total_tags*100:.0f}%)" for t, c in top_tags])
        top1_subject = subs[0] if subs else None
        top3_pct = top3_click_rate.get(sc, "—")
        long_tail_pct = long_tail_click_rate.get(sc, "—")
        return (sc, dtype, f"{t3_share:.0f}%", '、'.join(subs[:3]), top_tags_pct_str, top1_subject, top3_pct, long_tail_pct)
    # 按「需求-保存类型」分组：先输出高需求高保存（核心+商业化）合并一块，再输出高需求低保存
    table_rows = ''
    hi_save_strats = ['核心现金牛', '商业化优化']
    lo_save_strats = ['算法优化机会']
    # 高需求高保存：核心现金牛 + 商业化优化，第一列合并为一行
    hi_save_rows = []
    for strat in hi_save_strats:
        for sc in strat_groups.get(strat, []):
            hi_save_rows.append((strat, _scene_row_data(sc)))
    if hi_save_rows:
        rowspan_hi = len(hi_save_rows)
        save_type_td = f'<td rowspan="{rowspan_hi}" style="font-weight:600;text-align:center;vertical-align:middle">高需求<br>高保存</td>'
        for i, (strat, row) in enumerate(hi_save_rows):
            sc_name, dtype, t3_pct, subs_list, top_tags_pct_str, top1_subject, top3_click_pct, long_tail_pct = row
            is_first_of_strat = (i == 0 or hi_save_rows[i - 1][0] != strat)
            if is_first_of_strat:
                j = i
                while j < len(hi_save_rows) and hi_save_rows[j][0] == strat:
                    j += 1
                strat_span = j - i
                strat_td = f'<td rowspan="{strat_span}" style="color:{strat_colors[strat]};font-weight:700;text-align:center">{strat}</td>'
            else:
                strat_td = ''
            st_td = save_type_td if i == 0 else ''
            url = scene_subject_to_image.get((sc_name, top1_subject), '') or scene_to_image.get(sc_name, '')
            url = url_to_local.get(url, url) if url else ''
            if not isinstance(url, str):
                url = ''
            url_esc = url.replace('"', '&quot;').replace("'", '&#39;') if url else ''
            thumb = f'<span class="thumb-zoom" data-src="{url_esc}" onclick="window.thumbModalShow(this)"><img class="report-thumb" src="{url}" alt="" onerror="this.style.display=\'none\'" /></span>' if url else '—'
            table_rows += f"<tr>{st_td}{strat_td}<td>{sc_name}</td><td>{dtype}</td><td>{t3_pct}</td><td style=\"max-width:220px;min-width:140px;word-wrap:break-word;word-break:break-all\">{subs_list}</td><td>{thumb}</td><td>{top_tags_pct_str}</td><td>{top3_click_pct}</td><td>{long_tail_pct}</td></tr>"
    # 高需求低保存：算法优化机会
    algo_scenes_list = strat_groups.get('算法优化机会', [])
    if algo_scenes_list:
        rows_data = [_scene_row_data(sc) for sc in algo_scenes_list]
        rowspan_lo = len(rows_data)
        for i, row in enumerate(rows_data):
            sc_name, dtype, t3_pct, subs_list, top_tags_pct_str, top1_subject, top3_click_pct, long_tail_pct = row
            save_type_td = f'<td rowspan="{rowspan_lo}" style="font-weight:600;text-align:center;vertical-align:middle">高需求<br>低保存</td>' if i == 0 else ''
            strat_td = f'<td rowspan="{rowspan_lo}" style="color:{strat_colors["算法优化机会"]};font-weight:700;text-align:center">算法优化机会</td>' if i == 0 else ''
            url = scene_subject_to_image.get((sc_name, top1_subject), '') or scene_to_image.get(sc_name, '')
            url = url_to_local.get(url, url) if url else ''
            if not isinstance(url, str):
                url = ''
            url_esc = url.replace('"', '&quot;').replace("'", '&#39;') if url else ''
            thumb = f'<span class="thumb-zoom" data-src="{url_esc}" onclick="window.thumbModalShow(this)"><img class="report-thumb" src="{url}" alt="" onerror="this.style.display=\'none\'" /></span>' if url else '—'
            table_rows += f"<tr>{save_type_td}{strat_td}<td>{sc_name}</td><td>{dtype}</td><td>{t3_pct}</td><td style=\"max-width:220px;min-width:140px;word-wrap:break-word;word-break:break-all\">{subs_list}</td><td>{thumb}</td><td>{top_tags_pct_str}</td><td>{top3_click_pct}</td><td>{long_tail_pct}</td></tr>"
    cross_conclusion = f"""<table style="width:100%;border-collapse:collapse;font-size:13px;margin-left:20px;table-layout:fixed">
<thead><tr>
<th>需求-保存<br>类型</th><th>策略<br>类别</th><th>场景<br>名称</th><th>主体<br>集中度</th><th>TOP3主体<br>占比%</th><th style="max-width:220px;min-width:140px;word-wrap:break-word">主要<br>主体</th><th>TOP1主体<br>举例</th>
<th>复杂度标签<br>TOP3占比</th><th>TOP3主体<br>保存率%</th><th>长尾<br>保存率%</th>
</tr></thead><tbody>{table_rows}</tbody></table>"""

    # 修改为标题独立一行，标题下方两列图表
    c5b_html = ''
    def append_scene_group(title_color, title_text, charts):
        if not charts:
            return
        c5b_html_local = f'<h4 style="font-weight:700;color:{title_color};margin:8px 0 4px;font-size:14px">{title_text}</h4>'
        chart_pairs = [charts[i:i+2] for i in range(0,len(charts),2)]
        for pair in chart_pairs:
            if len(pair)==2 and pair[0] and pair[1]:
                c5b_html_local += f'<div class="grid-2"><div class="chart-wrap">{pair[0]}</div><div class="chart-wrap">{pair[1]}</div></div>'
            else:
                for ch in pair:
                    if ch:
                        c5b_html_local += f'<div class="chart-wrap">{ch}</div>'
        return c5b_html_local

    core_charts = [chart_s5_sub(df, sc) for sc in core_scenes if chart_s5_sub(df, sc)]
    biz_charts = [chart_s5_sub(df, sc) for sc in biz_opt_scenes if chart_s5_sub(df, sc)]
    algo_charts = [chart_s5_sub(df, sc) for sc in algo_scenes if chart_s5_sub(df, sc)]

    core_biz_html = (append_scene_group("#16a34a", "核心现金牛场景", core_charts) or '') + (append_scene_group("#3b82f6", "商业化优化场景", biz_charts) or '')
    c5b_html += '<details style="margin-bottom:12px"><summary>核心现金牛场景 + 商业化优化场景（点击展开）</summary>' + core_biz_html + '</details>'
    c5b_html += append_scene_group("#ef4444", "算法优化机会场景", algo_charts) or ''

    # ── 新老用户蝴蝶图（4个：占比、点击率、付费率、新增付费率） ──
    # 场景范围与其他模块一致：占比>=1% 且非"未确定"
    bf_scenes = gd.sort_values('count', ascending=False)['scene_l2'].tolist()
    c6_share = _butterfly_rate(df, '场景占比(%)', '场景占比', max_val=None, scene_list=bf_scenes)
    c6_click = _butterfly_rate(df, '下载点击率(%)', '下载点击率', max_val=100, scene_list=bf_scenes)
    c6_save  = _butterfly_rate(df, '下载成功率(%)', '下载成功率（付费率）', max_val=100, scene_list=bf_scenes)
    c6_pay   = _butterfly_rate(df, '新增付费率(%)', '新增付费率', max_val=None, scene_list=bf_scenes)
    c6_loss  = _butterfly_rate(df, '流失率(%)', '下载点击到下载成功流失率', max_val=100, scene_list=bf_scenes)

    # ── 策略气泡图 ──
    c7_html, med_x, med_y, med_z = chart_s7_bubble(gd)

    # ── 数据素材 ──
    big_low = g_no_undet[(g_no_undet['count']>=50)&(g_no_undet['下载点击率(%)']<65)]
    big_low_list = '、'.join(big_low['scene_l2'].tolist()) if not big_low.empty else '无'
    hi_click_top = g_no_undet[g_no_undet['count']>=15].nlargest(3,'下载点击率(%)')
    # 商业化转化结论仅基于占比≥1% 场景（与图表 c3 一致）
    hi_save = gd.nlargest(3,'下载成功率(%)')
    lo_save = gd.nsmallest(3,'下载成功率(%)')
    hi_pay  = gd[gd['新增付费率(%)']>0].nlargest(3,'新增付费率(%)')
    s3_hi_save_parts = [f"{row['scene_l2']} {row['下载成功率(%)']:.1f}%" for _, row in hi_save.iterrows()]
    s3_hi_save_text = "、".join(s3_hi_save_parts) if s3_hi_save_parts else "—"
    s3_lo_save_parts = [f"{row['scene_l2']} {row['下载成功率(%)']:.1f}%" for _, row in lo_save.iterrows()]
    s3_lo_save_text = "、".join(s3_lo_save_parts) if s3_lo_save_parts else "—"
    s3_hi_pay_parts = [f"{row['scene_l2']} {row['新增付费率(%)']:.2f}%" for _, row in hi_pay.iterrows()]
    s3_hi_pay_text = "、".join(s3_hi_pay_parts) if s3_hi_pay_parts else "—"

    hi_loss = g_no_undet[(g_no_undet['count']>=15)&(g_no_undet['click_count']>0)].nlargest(3,'流失率(%)')
    big_block = g_no_undet[(g_no_undet['下载点击率(%)']>=70)&(g_no_undet['流失率(%)']>=35)].sort_values('count',ascending=False).head(3)['scene_l2'].tolist()
    top3_hi_download = '、'.join(hi_click_top['scene_l2'].tolist()) if not hi_click_top.empty else '—'

    gs1 = df.dropna(subset=['subject_l1']).groupby('subject_l1').agg(
        count=('device_id','count'),click_count=('is_download_click','sum'),saved_count=('is_saved','sum')).reset_index()
    gs1['占比(%)']=(gs1['count']/gs1['count'].sum()*100).round(1)
    gs1['下载点击率(%)']=(gs1['click_count']/gs1['count']*100).round(1)
    gs1['下载成功率(%)']=(gs1['saved_count']/gs1['click_count'].replace(0,1)*100).round(1)
    gs1=gs1.sort_values('count',ascending=False).reset_index(drop=True)
    gs1d=gs1[['subject_l1','count','占比(%)','click_count','下载点击率(%)','下载成功率(%)']].rename(
        columns={'subject_l1':'主体L1','count':'需求量','click_count':'点击人数'})
    sub_top3=gs1.head(3)['subject_l1'].tolist()
    sub_hi=gs1.nlargest(3,'下载点击率(%)'); sub_lo=gs1.nsmallest(3,'下载点击率(%)')
    # 每个举例主体对应各自数据（不用 max/min 混用）
    demand_top3_parts = [f"{row['subject_l1']} {row['占比(%)']:.1f}%" for _, row in gs1.head(3).iterrows()]
    subject_demand_text = "、".join(demand_top3_parts) if demand_top3_parts else "—"
    click_hi_parts = [f"{row['subject_l1']} {row['下载点击率(%)']:.1f}%" for _, row in sub_hi.iterrows()]
    subject_click_hi_text = "、".join(click_hi_parts) if click_hi_parts else "—"
    click_lo_parts = [f"{row['subject_l1']} {row['下载点击率(%)']:.1f}%" for _, row in sub_lo.iterrows()]
    subject_click_lo_text = "、".join(click_lo_parts) if click_lo_parts else "—"

    # 二级主体：需求占比与下载点击率（附录用）
    gs2 = df.dropna(subset=['subject_l1', 'subject_l2']).groupby('subject_l2').agg(
        count=('device_id', 'count'), click_count=('is_download_click', 'sum'), subject_l1=('subject_l1', 'first')).reset_index()
    total2 = gs2['count'].sum() or 1
    gs2['需求占比(%)'] = (gs2['count'] / total2 * 100).round(1)
    gs2['下载点击率(%)'] = (gs2['click_count'] / gs2['count'].replace(0, 1) * 100).round(1)
    gs2 = gs2.sort_values('count', ascending=False).reset_index(drop=True)
    demand_l2_top = [f"{row['subject_l2']}（{row['subject_l1']}）{row['需求占比(%)']:.1f}%" for _, row in gs2.head(3).iterrows()]
    subject_l2_demand_text = "、".join(demand_l2_top) if demand_l2_top else "—"
    click_lo_l2 = gs2.nsmallest(3, '下载点击率(%)')
    click_lo_l2_parts = [f"{row['subject_l2']}（{row['subject_l1']}）{row['下载点击率(%)']:.1f}%" for _, row in click_lo_l2.iterrows()]
    subject_l2_click_lo_text = "、".join(click_lo_l2_parts) if not click_lo_l2.empty else "—"
    subject_l2_conclusion_text = f"1. <strong>需求占比最高的二级主体</strong>：{subject_l2_demand_text}。<br>2. <strong>下载点击率最低</strong>：{subject_l2_click_lo_text}。"
    c5a_l2 = chart_s5_subject_l2(df)

    gu = df.dropna(subset=['user_type']).groupby('user_type').agg(
        count=('device_id','count'),click=('is_download_click','sum'),
        saved=('is_saved','sum'),pay=('paid_for_url','sum')).reset_index()
    nr=gu[gu['user_type']=='新用户']; orr=gu[gu['user_type']=='老用户']
    ncr=int(nr['click'].values[0])/int(nr['count'].values[0])*100 if not nr.empty else 0
    ocr=int(orr['click'].values[0])/int(orr['count'].values[0])*100 if not orr.empty else 0
    nsr=int(nr['saved'].values[0])/int(nr['count'].values[0])*100 if not nr.empty else 0
    osr=int(orr['saved'].values[0])/int(orr['count'].values[0])*100 if not orr.empty else 0
    npay=int(nr['pay'].values[0]) if not nr.empty else 0
    opay=int(orr['pay'].values[0]) if not orr.empty else 0
    gu_scene = df.dropna(subset=['scene_l2','user_type'])
    gu_scene = gu_scene[gu_scene['scene_l2']!='未确定'].groupby(['scene_l2','user_type']).size().reset_index(name='cnt')
    new_top3 = gu_scene[gu_scene['user_type']=='新用户'].nlargest(3,'cnt')['scene_l2'].tolist()
    old_top3 = gu_scene[gu_scene['user_type']=='老用户'].nlargest(3,'cnt')['scene_l2'].tolist()

    # 表格（仅展示 gd 范围的数据）
    gd_tbl = gd[['scene_l2','count','占比(%)','下载成功率(%)','新增付费率(%)','paid_count']].copy()
    gd_tbl = gd_tbl.rename(columns={'scene_l2':'Scene L2','count':'需求量','下载成功率(%)':'下载成功率（下载成功/下载点击）(%)','paid_count':'新增付费（人）'})
    pay_tbl = _tbl(gd_tbl, hl='新增付费率(%)')
    gd_loss = gd[gd['click_count']>0][['scene_l2','count','下载点击率(%)','下载成功率(%)','流失率(%)','新增付费率(%)']].sort_values('流失率(%)',ascending=False)
    loss_tbl = _tbl(gd_loss.rename(columns={'scene_l2':'Scene L2','count':'需求量'}), hl='流失率(%)', hg=False)
    g1d = g.head(20)[['scene_l2','count','user_count','占比(%)','下载点击率(%)','下载成功率(%)','新增付费率(%)']].rename(
        columns={'scene_l2':'Scene L2','count':'抠图人数','user_count':'去重人数'})
    tbl_s1 = _tbl(g1d, hl='抠图人数')

    # 策略分布（三维六类：需求+点击率+付费率）
    def _st_label(r):
        big=r['占比(%)']>=med_x; good=r['下载点击率(%)']>=med_y; pay=r['下载成功率(%)']>=med_z
        if big and good and pay: return '核心现金牛'
        if big and good and not pay: return '商业化优化'
        if big and not good: return '算法优化机会'
        if not big and good and pay: return '小众高价值'
        if not big and good and not pay: return '小众优质'
        return '低优先级'
    g15=gd.copy(); g15['策略']=g15.apply(_st_label,axis=1)
    strat_dist = g15['策略'].value_counts().to_dict()
    algo_scenes = g15[g15['策略']=='算法优化机会']['scene_l2'].tolist()
    core_scenes = g15[g15['策略']=='核心现金牛']['scene_l2'].tolist()
    biz_opt_scenes = g15[g15['策略']=='商业化优化']['scene_l2'].tolist()
    niche_hi_scenes = g15[g15['策略']=='小众高价值']['scene_l2'].tolist()
    niche_scenes = g15[g15['策略']=='小众优质']['scene_l2'].tolist()

    # ── 组装 HTML ──
    toc_html = """
<nav id="toc" class="toc collapsed">
  <div class="toc-head"><span>报告目录</span><button class="toc-close" onclick="tocToggle(false)">✕</button></div>
  <a href="#sec-summary">核心发现与行动建议</a>
  <a href="#sec-kpi">关键指标概览</a>
  <a href="#sec-1">一、场景分析</a>
  <a href="#sec-2">二、主体分析</a>
  <a href="#sec-3">三、新老用户差异</a>
  <a href="#sec-appendix">附录：抠图打标分类体系</a>
</nav>
<button class="toc-open" onclick="tocToggle(true)">☰ 目录</button>"""

    appendix_subject_block = f'''<div class="card" style="margin-top:16px"><h3>二、主体分析：哪类主体抠图效果好/差？</h3>
  <div class="conclusion"><div class="ttl">分析结论 — 一级主体</div>
  <p>1. <strong>需求最大主体</strong>：{subject_demand_text}。<br>
  2. <strong>下载点击率最高</strong>：{subject_click_hi_text}。<br>
  3. <strong>下载点击率最低</strong>：{subject_click_lo_text}。</p></div>
</div>
<div class="chart-scroll-x"><div class="chart-wrap">{c5a}</div></div>
<details><summary>展开一级主体指标表</summary>{_tbl(gs1d, hl='下载点击率(%)')}</details>
<div class="card" style="margin-top:16px"><h3>三、二级主体：需求占比与下载点击率</h3>
  <div class="conclusion"><div class="ttl">分析结论</div>
  <p>{subject_l2_conclusion_text}</p></div>
<div class="chart-scroll-x"><div class="chart-wrap">{c5a_l2}</div></div>
</div>'''
    appendix_html = build_classification_appendix(extra_after_part1=appendix_subject_block)
    new_old_user_table_html = build_new_old_user_table()
    _strategy_html = strategy_table(gd, med_x, med_y, med_z, scene_to_image, url_to_local)
    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>抠图场景机会分析报告</title>{ECHARTS_JS}{CSS}</head><body>
<div class="hero-top-line"></div>
{toc_html}
<div class="hero">
  <h1>抠图场景机会分析报告</h1>
  <div class="meta">
    <span>数据日期：2026-03-04</span><span>随机抽取样本2000条，有效样本：{N:,} 条</span>
    <span>核心粒度：Scene L2（{len(g)}个场景，展示≥1%的{len(gd)}个）</span>
    <span>新用户 {n_new} / 老用户 {n_old}</span>
  </div>
  <div class="meta" id="sec-kpi">
    <span>下载点击率：{tc/N*100:.1f}%（{tc:,}人点击）</span><span>下载成功率：{avg_save_rate:.1f}%（各场景均值）</span><span>新增付费率：{tp/tc_safe*100:.2f}%（{tp}人新增付费）</span>
  </div>
</div>
<div class="container">

<!-- 执行摘要 -->
<div class="exec-summary no-highlight" id="sec-summary">
<h3>核心发现与行动建议</h3>
<p style="font-size:14px;line-height:1.75;color:#334155;margin-bottom:20px">抠图需求高度集中，TOP10 占 <b>{top10_share:.0f}%</b>，TOP3 主要为电商场景 {top3[0]}、{top3[1]}、{top3[2]}。</p>
<div class="grid">
  <div class="exec-item" style="background:#fee2e2">
    <div class="tag" style="color:#991b1b">算法优化（效果算法侧）</div>
    <p style="font-size:13px;line-height:1.6;margin:0">
      · <span class="item-ttl">主要优化场景</span>：{'、'.join(algo_scenes) if algo_scenes else '无'}<br>
      · <span class="item-ttl">主要优化主体</span>：公章/印章，上装(T恤/衬衫/卫衣/外套/马甲)，医疗用品(药品/保健品/医疗器具)，家具(沙发/椅子/桌子/床垫/柜子)<br>
      · <span class="item-ttl">优化方向</span>：对<b>多主体</b>场景加强分割与主体识别，增强商品图<b>普适性</b>策略<br>
      · <span class="item-ttl">待研究</span>：反光高光的阴影影响
    </p>
  </div>
  <div class="exec-item" style="background:#fff7ed">
    <div class="tag" style="color:#9a3412">商业化优化（产品、商业化侧）</div>
    <p style="font-size:13px;line-height:1.6;margin:0">
      · <span class="item-ttl">主要优化场景</span>：<br>
      <span class="sub-ttl">高需求+高保存</span>：平面拼贴/合成、Logo/图标提取；<br>
      <span class="sub-ttl">低需求+高保存</span>：印花/图案设计、菜单/菜品展示、广告横幅；<br>
      · <span class="item-ttl">优化方向</span>：<br>
      <span class="sub-ttl">高需求高保存场景</span>：做<b>用户访谈</b>做进一步归因，激活付费；<br>
      <span class="sub-ttl">小众高价值场景</span>：做<b>产品延展，定制化付费</b>（如菜品精修、付费菜单模版素材等），挖掘存量价值。
    </p>
  </div>
  <div class="exec-item" style="background:#eff6ff">
    <div class="tag" style="color:#1e40af">用户增长（运营、用户增长侧）</div>
    <p style="font-size:13px;line-height:1.6;margin:0">
      · <span class="item-ttl">主要场景</span>：Logo/图标提取，商品去底/白底图、社交配图<br>
      · <span class="item-ttl">优化方向</span>：<br>
      <span class="sub-ttl">拉新</span>：聚焦 <b>Logo/图标提取</b>的运营策略；<br>
      <span class="sub-ttl">强化</span>：商品去底/白底图、社交配图（<b>换背景和 C 端人像</b>）重点推广和强化，巩固存量并带动整体规模。
    </p>
  </div>
</div>
</div>

<!-- 一、场景分析 -->
<div class="section" id="sec-1">
<h2>一、场景分析</h2>
<div class="insight plain" style="margin-bottom:16px"><div class="ttl">指标口径</div>
<p>· <b>有效抠图记录</b>：样本中的有效记录数。<br>
· <b>需求占比(%)</b>：该场景抠图人数 / 总抠图人数 × 100。<br>
· <b>下载点击率(%)</b>：下载点击数 / 该场景抠图人数 × 100（整体为 下载点击数/有效记录数×100）。<br>
· <b>下载成功率(%)</b>：单场景为 下载成功数/下载点击数×100（下载成功即已付费）<br>
· <b>新增付费率(%)</b>：新增付费数 / 下载点击数 × 100；口径：用户在<b>抠完当前图片后、抠下一张图之前</b>是否付费。<br>
· <b>流失率(%)</b>：(1 − 下载成功数/下载点击数)×100，即从「点击下载」到「下载成功」的流失比例。<br>
· <b>策略地图</b>：横轴=下载点击率，纵轴=付费转化率，气泡大小=场景占比；以各自中位数为分界线划分六类策略，仅展示占比≥1% 场景。</p></div>
<h3 style="margin:16px 0 8px">1.1 所有场景需求占比</h3>
<div class="conclusion" style="margin-bottom:16px"><div class="ttl">核心观察</div>
<p>需求高度集中：TOP10 场景占 <b>{top10_share:.1f}%</b>，TOP3主要为电商场景（{top3[0]}、{top3[1]}、{top3[2]}）合计 {g.head(3)['占比(%)'].sum():.1f}%。<br>
{len(g[g['占比(%)']<1])} 个场景需求占比小于1%，数据量较小，本次分析不做深入研究。</p></div>
<div class="chart-scroll-x"><div class="chart-wrap">{c1}</div></div>
<h3 style="margin:16px 0 8px">1.2 场景策略地图</h3>
<div class="conclusion" style="margin-bottom:16px"><div class="ttl">策略地图结论</div>
<p>· <strong>核心现金牛（{strat_dist.get('核心现金牛',0)}个）</strong>：{'、'.join(core_scenes[:5]) if core_scenes else '无'}——需求大、效果好、付费高，<b>重点推广和强化</b>。<br>
· <strong>商业化优化（{strat_dist.get('商业化优化',0)}个）</strong>：{'、'.join(biz_opt_scenes[:5]) if biz_opt_scenes else '无'}——需求大，效果获得认可但付费率低于中位数，可通过<b>用户访谈</b>了解用户不愿付费的原因，是价格门槛的原因还是低频需求导致不愿长期订阅<br>
· <strong>算法优化机会（{strat_dist.get('算法优化机会',0)}个）</strong>：{'、'.join(algo_scenes[:5]) if algo_scenes else '无'}——需求大但下载点击率低，优先做<b>算法效果优化</b>，再进一步推广。其中电商海报/活动页、文档配图保存率最低，可对标保存率超 80% 的Logo/图标提取、企业/职场头像、菜单/菜品展示作为效果标杆。<br>
· <strong>小众高价值（{strat_dist.get('小众高价值',0)}个）</strong>：{'、'.join(niche_hi_scenes[:5]) if niche_hi_scenes else '无'}——需求小但用户满意度和付费率双高，可做<b>溢价定价或差异化拓展</b>，比如对应场景的付费素材模版<br>
· <strong>小众优质（{strat_dist.get('小众优质',0)}个）</strong>：{'、'.join(niche_scenes[:5]) if niche_scenes else '无'}——下载率高但需求小付费低，<b>维持现状</b>，未来根据用户反馈再决定是否商业化激活或用户增长。<br>
· <strong>低优先级（{strat_dist.get('低优先级',0)}个）</strong>：需求小且效果差，暂不投入。</p></div>
<div class="chart-wrap">{c7_html}</div>
<div class="conclusion" style="margin-bottom:16px"><div class="ttl">其他观察</div>
<p><strong>2、拦截损失</strong><br>
<strong>高付费流失场景（文档配图和证件照）</strong>：文档配图核心问题是下载点击率低（体验/算法问题），而非商业化阻力，因此暂不做商业化优化，优先提升下载转化。证件照为低频一次性需求，订阅付费意愿天然弱，需求占比小，商业化价值有限，暂不投入资源。<em>结论：两类场景优先聚焦各自核心问题。</em><br>
<strong>中付费流失场景中的 Logo/图标提取和平面拼贴/合成</strong>：需求占比高、下载点击率高，说明用户已高度认可产品效果，仅在付费环节存在中度流失，是当前最具付费激活潜力的核心增量场景，可作为商业化优化的重点。</p></div>
<details style="margin-bottom:16px"><summary>策略分类逻辑（三维六类）</summary>
<div class="card" style="margin-top:8px">
  <p style="font-size:14px;color:var(--muted)">横轴=下载点击率，纵轴=付费转化率，气泡大小=场景占比；以各自中位数为分界线，结合三个维度划分 6 类策略。</p>
  <table>
    <tr style="background:#f8fafc"><th style="padding:8px">需求</th><th style="padding:8px">下载点击率</th><th style="padding:8px">付费率</th><th style="padding:8px">定位</th><th style="padding:8px">策略</th></tr>
    <tr><td>大</td><td>高</td><td>高</td><td style="background:#dcfce7;font-weight:600">核心现金牛</td><td>重点强化 & 推广</td></tr>
    <tr><td>大</td><td>高</td><td>低</td><td style="background:#dbeafe;font-weight:600">商业化优化</td><td>效果好，激活付费转化</td></tr>
    <tr><td>大</td><td colspan="2">低（不分付费）</td><td style="background:#fee2e2;font-weight:600">算法优化机会</td><td>算法/体验优化</td></tr>
    <tr><td>小</td><td>高</td><td>高</td><td style="background:#fef3c7;font-weight:600">小众高价值</td><td>溢价定价/差异化拓展</td></tr>
    <tr><td>小</td><td>高</td><td>低</td><td style="background:#ede9fe;font-weight:600">小众优质</td><td>维持/尝试商业化</td></tr>
    <tr><td>小</td><td colspan="2">低（不分付费）</td><td style="background:#f1f5f9;color:#6b7280">低优先级</td><td>暂不投入</td></tr>
  </table>
  <p style="font-size:12px;color:var(--muted);margin-top:8px">中位数分界线：需求占比 {med_x:.2f}%，下载点击率 {med_y:.0f}%，付费转化率 {med_z:.0f}%。仅展示占比≥1%场景。</p>
</div>
</details>
<h3 style="margin:16px 0 8px">策略矩阵明细</h3>
{_strategy_html}
<details style="margin-top:12px"><summary>展开流失率明细表</summary>{loss_tbl}</details>
</div>

<!-- 二、主体分析 -->
<div class="section" id="sec-2">
<h2>二、主体分析</h2>
  <div class="card">
  <h3>场景×主体交叉分析 & 复杂度标签词云</h3>
  <p style="color:var(--muted);font-size:13px;margin-bottom:8px">高需求高保存（核心现金牛+商业化优化） vs 高需求低保存（算法优化机会）</p>
  <p style="font-size:16px;font-weight:700;color:var(--sub);margin:12px 0 8px">高需求场景中：高低保存场景的主体分布和技术难度有何差异？</p>
  <div class="conclusion"><div class="ttl">交叉分析结论</div>
  <p>{cross_conclusion_text}</p>
  </div>
  <h4 style="font-size:14px;font-weight:700;margin:16px 0 8px;color:var(--sub)">交叉分析明细表</h4>
  <div class="tbl-wrap tbl-wrap--sticky-head">{cross_conclusion}</div>
  <h3 style="margin:16px 0 8px">复杂度标签 词云对比</h3>
  <div class="conclusion" style="margin-bottom:12px"><div class="ttl">结论</div><p><b>多主体</b>在<b>低保存场景</b>占比更高，是导致保存率偏低的原因之一。<br>
反光/高光 在低保存场景占比较大，需要进一步研究是否受阴影带来的负面影响。<br>
而<b>复杂边缘</b>在<b>高保存场景</b>占比更高，可能并非拉低保存率的主要因素。</p></div>
  <div class="grid-2 wc-equal-height">{wc_core_block}{wc_algo_block}</div>
  {wc_tag_examples_card}
  </div>
  <div class="card" style="margin-top:16px">
  <h3>主体优化方向</h3>
  <p style="font-size:13px;color:var(--muted);margin-bottom:8px">算法优化主体占大盘与投入建议（场景内保存率口径=下载点击率，按占大盘比值降序）</p>
  {algo_subject_table_html}
  </div>
<div class="chart-scroll-x cross-analysis-charts">
{c5b_html}
</div>
</div>

<!-- 三、新老用户分析 -->
<div class="section" id="sec-3">
<h2>三、新老用户分析</h2>
<div class="card">
  <h3>新老用户需求是否不同？转化行为是否不同？</h3>
  <div class="conclusion"><div class="ttl">分析结论</div>
  <p><strong>（一）新用户拉新方向</strong><br>
  Logo / 图标提取：高下载 + 高首单优势，可针对该场景做运营投放拉新；<br>
  文档配图：优先完成图章印章算法优化，解决抠图效果痛点<br>
  <br>
  <strong>（二）老用户激活方向</strong><br>
  付费激活： Logo / 图标提取、企业 / 职场头像场景，老用户下载点击率高，但是付费流失率较其他场景更高，可进一步用户访谈了解流失原因，从而针对性制定付费激活策略；<br>
  小众场景商业化：为菜单 / 菜品展示场景设计定制化付费方案，比如付费菜品精修等延展功能，挖掘存量会员价值。</p></div>
</div>
<h4 style="margin-top:20px;margin-bottom:8px">新老用户场景占比与核心判断</h4>
{new_old_user_table_html}
<p style="font-size:13px;color:var(--muted);margin-bottom:8px">蝴蝶图：← 左侧新用户 / 右侧老用户 →，横轴 0 居中</p>
<div class="chart-scroll-x"><div class="grid-2">
  <div class="chart-wrap">{c6_share}</div>
  <div class="chart-wrap">{c6_click}</div>
</div></div>
<div class="chart-scroll-x"><div class="grid-2" style="margin-top:12px">
  <div class="chart-wrap">{c6_save}</div>
  <div class="chart-wrap">{c6_pay}</div>
</div></div>
</div>

</div>

<div class="section" id="sec-appendix">
<h2>附录：抠图打标分类体系</h2>
<p style="font-size:14px;color:var(--muted)">本报告中的场景、主体及复杂度标签均基于以下分类体系（v4），便于理解口径与子类含义。</p>
{appendix_html}
</div>

{TOC_JS}
</body></html>"""

    out = 'statistics/full_analysis_report.html'
    with open(out,'w',encoding='utf-8') as f: f.write(html)
    print(f"\n✅ 报告已生成: {out}  ({len(html)//1024} KB)")

if __name__ == '__main__':
    main()
