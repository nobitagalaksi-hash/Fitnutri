"""FitNutri AI - workout harian & nutrisi berbasis Generative AI (Gemini API, ringan, tanpa GPU)."""
import html
import json
import os
from datetime import date

import gradio as gr
import pandas as pd
from pydantic import BaseModel

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
FALLBACK = os.getenv("GEMINI_FALLBACK", "gemini-3.1-flash-lite")
ENV_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
DEMO = os.getenv("DEMO_MODE") == "1"

SYSTEM = (
    "Kamu adalah ahli gizi dan pelatih kebugaran berbahasa Indonesia. Saran harus aman, praktis, "
    "dan realistis, prioritaskan makanan Indonesia. Gunakan angka target dari pengguna, jangan "
    "mengarang angka target. Ingatkan konsultasi ke dokter bila ada kondisi medis. Jawab ringkas."
)


# ------------------------------------------------------------------ SKEMA OUTPUT AI
class Item(BaseModel):
    nama: str
    porsi: str          # takaran, mis. "1 piring", "2 potong"
    gram: float         # estimasi berat porsi (g)
    kalori: float
    protein_g: float
    karbo_g: float
    lemak_g: float
    serat_g: float
    gula_g: float
    natrium_mg: float


class ScanResult(BaseModel):
    makanan: list[Item]
    skor_gizi: int      # 1-10
    catatan: str
    saran: str
    alergen: list[str]  # alergen/bahan sensitif yang kemungkinan ada


# ------------------------------------------------------------------ PANGGIL GEMINI
def _client(key):
    k = (key or ENV_KEY or "").strip()
    if not k:
        raise gr.Error("API key Gemini belum diisi. Buka tab ⚙️ Pengaturan (gratis di aistudio.google.com).")
    from google import genai
    return genai.Client(api_key=k)


def _cfg(schema, temperature):
    from google.genai import types
    kw = dict(system_instruction=SYSTEM, temperature=temperature)
    if schema:
        kw.update(response_mime_type="application/json", response_schema=schema)
    return types.GenerateContentConfig(**kw)


def _err(e):
    s = str(e)
    if "429" in s or "RESOURCE_EXHAUSTED" in s:
        return "Kuota gratis Gemini sedang habis, coba lagi beberapa menit."
    if "API key" in s or "API_KEY" in s or "401" in s or "403" in s:
        return "API key tidak valid."
    return s[:200]


def generate(contents, schema=None, temperature=0.3, key=None):
    c, last = _client(key), None
    for m in dict.fromkeys([MODEL, FALLBACK]):
        try:
            return c.models.generate_content(model=m, contents=contents, config=_cfg(schema, temperature)).text
        except Exception as e:  # noqa: BLE001
            last = e
    raise gr.Error(f"Gagal memanggil AI: {_err(last)}")


def stream(contents, temperature=0.5, key=None):
    if DEMO:
        acc = ""
        for w in "**[MODE DEMO]** Jawaban AI muncul di sini, lengkap dengan saran praktis.".split():
            acc += w + " "
            yield acc
        return
    c, last = _client(key), None
    for m in dict.fromkeys([MODEL, FALLBACK]):
        acc = ""
        try:
            for ch in c.models.generate_content_stream(model=m, contents=contents, config=_cfg(None, temperature)):
                acc += ch.text or ""
                yield acc
            return
        except Exception as e:  # noqa: BLE001
            last = e
            if acc:
                return
    raise gr.Error(f"Gagal memanggil AI: {_err(last)}")


DEMO_SCAN = {
    "makanan": [
        {"nama": "Nasi putih", "porsi": "1 piring", "gram": 150, "kalori": 195, "protein_g": 4, "karbo_g": 43,
         "lemak_g": 0.4, "serat_g": 0.6, "gula_g": 0, "natrium_mg": 2},
        {"nama": "Ayam goreng", "porsi": "1 paha", "gram": 90, "kalori": 260, "protein_g": 22, "karbo_g": 6,
         "lemak_g": 16, "serat_g": 0.2, "gula_g": 0, "natrium_mg": 420},
        {"nama": "Tempe goreng", "porsi": "2 potong", "gram": 50, "kalori": 110, "protein_g": 7, "karbo_g": 6,
         "lemak_g": 7, "serat_g": 1.5, "gula_g": 0, "natrium_mg": 8},
    ],
    "skor_gizi": 6, "catatan": "Protein cukup, serat rendah dan lemak dari gorengan.",
    "saran": "Tambahkan lalapan atau sayur, kurangi gorengan, ganti sebagian nasi dengan sayur.",
    "alergen": ["kedelai"],
}


# ------------------------------------------------------------------ LOGIKA GIZI
AKTIVITAS = {"Jarang olahraga": 1.2, "1-3x/minggu": 1.375, "3-5x/minggu": 1.55, "6-7x/minggu": 1.725}
TUJUAN = ["Turunkan berat badan", "Naikkan massa otot", "Jaga berat badan"]
KEYS = ["kalori", "protein_g", "karbo_g", "lemak_g", "serat_g", "gula_g", "natrium_mg"]
COLS = ["Makanan", "Porsi (g)", "Kalori", "Protein", "Karbo", "Lemak", "Serat", "Gula", "Natrium (mg)"]


def hitung_target(nama, gender, usia, berat, tinggi, aktivitas, tujuan, level, hari, alat, pantangan):
    bmr = 10 * berat + 6.25 * tinggi - 5 * usia + (5 if gender == "Pria" else -161)
    tdee = bmr * AKTIVITAS[aktivitas]
    kal = tdee - 400 if tujuan == TUJUAN[0] else tdee + 300 if tujuan == TUJUAN[1] else tdee
    protein = berat * 1.8
    lemak = kal * 0.25 / 9
    karbo = (kal - protein * 4 - lemak * 9) / 4
    bmi = berat / (tinggi / 100) ** 2
    t = dict(kalori=round(kal), protein=round(protein), karbo=round(karbo), lemak=round(lemak),
             bmi=round(bmi, 1), tdee=round(tdee), bmr=round(bmr))
    profil = dict(nama=nama, gender=gender, usia=usia, tujuan=tujuan, level=level, hari=hari,
                  alat=alat, pantangan=pantangan or "", target=t)
    return profil, kartu_target(t)


def bmi_kat(b):
    return "Kurus" if b < 18.5 else "Normal" if b < 25 else "Berat berlebih" if b < 30 else "Obesitas"


def kartu_target(t):
    return f"""<div class="stats">
<div class="stat big"><span>Target kalori</span><b>{t['kalori']}</b><small>kkal / hari</small></div>
<div class="stat"><span>Protein</span><b>{t['protein']} g</b></div>
<div class="stat"><span>Karbohidrat</span><b>{t['karbo']} g</b></div>
<div class="stat"><span>Lemak</span><b>{t['lemak']} g</b></div>
<div class="stat"><span>BMI</span><b>{t['bmi']}</b><small>{bmi_kat(t['bmi'])}</small></div>
<div class="stat"><span>TDEE</span><b>{t['tdee']}</b><small>BMR {t['bmr']}</small></div></div>"""


def rencana(profil, key):
    if not profil:
        raise gr.Error("Klik 'Hitung Target' dulu.")
    t = profil["target"]
    pw = (f"Buat jadwal workout 7 hari untuk {profil['gender']}, usia {int(profil['usia'])}, level {profil['level']}, "
          f"tujuan {profil['tujuan']}, {int(profil['hari'])} hari latihan/minggu, alat: {profil['alat'] or 'tanpa alat'}. "
          "Format per hari: sesi, gerakan, set x repetisi, durasi. Sertakan hari istirahat dan tips pemanasan.")
    pm = (f"Buat menu makan 1 hari (sarapan, siang, malam, 2 snack) makanan Indonesia. Target {t['kalori']} kkal, "
          f"protein {t['protein']} g, karbo {t['karbo']} g, lemak {t['lemak']} g. Pantangan/alergi: "
          f"{profil['pantangan'] or 'tidak ada'}. Tampilkan estimasi kalori dan protein tiap waktu makan.")
    yield "_Menyusun jadwal workout..._", "_Menunggu..._"
    w = ""
    for w in stream(pw, key=key):
        yield w, "_Menunggu..._"
    for m in stream(pm, key=key):
        yield w, m


# ------------------------------------------------------------------ SCAN NUTRISI
def _kecilkan(img):
    img = img.convert("RGB")
    img.thumbnail((1024, 1024))
    return img


def _prompt(mode, hint, profil):
    extra = f"\nInfo dari pengguna: {hint}" if hint else ""
    alergi = f"\nPantangan/alergi pengguna: {profil['pantangan']}" if profil and profil.get("pantangan") else ""
    if mode == "Label kemasan":
        base = ("Baca tabel Informasi Nilai Gizi pada foto label kemasan ini. Satu item per produk, gunakan "
                "nilai PER SATU KEMASAN/TOTAL yang dikonsumsi (kalikan jumlah sajian bila perlu, sebutkan di 'porsi'). "
                "Isi gram dari berat bersih/takaran saji. skor_gizi menilai kualitas produk (gula, natrium, lemak jenuh).")
    elif mode == "Deskripsi teks":
        base = "Estimasi nutrisi dari deskripsi makanan berikut, pecah per item makanan."
    else:
        base = ("Identifikasi SEMUA makanan/minuman pada foto, estimasi berat tiap item (gram) dan nutrisinya. "
                "Prioritaskan makanan Indonesia. Jika bukan makanan, kembalikan daftar makanan kosong dan jelaskan di catatan.")
    return base + " Semua angka bertipe angka. Saran: 1-2 kalimat perbaikan gizi yang konkret." + alergi + extra


def scan(foto, mode, deskripsi, hint, profil, key):
    if mode == "Deskripsi teks":
        if not (deskripsi or "").strip():
            raise gr.Error("Tulis dulu makanan yang kamu makan.")
        contents = [_prompt(mode, hint, profil) + "\nDeskripsi: " + deskripsi]
    else:
        if foto is None:
            raise gr.Error("Upload atau ambil foto dulu.")
        contents = [_kecilkan(foto), _prompt(mode, hint, profil)]
    if DEMO:
        data = DEMO_SCAN
    else:
        raw = generate(contents, schema=ScanResult, temperature=0.2, key=key)
        try:
            data = json.loads(raw)
        except Exception:  # noqa: BLE001
            raise gr.Error("Hasil AI tidak terbaca, coba ulangi dengan foto yang lebih jelas.")
    items = [dict(i) for i in data.get("makanan", [])]
    if not items:
        raise gr.Error("Tidak ada makanan terdeteksi: " + str(data.get("catatan", "")))
    meta = dict(skor=data.get("skor_gizi", 0), catatan=data.get("catatan", ""),
                saran=data.get("saran", ""), alergen=data.get("alergen", []))
    return items, meta, to_df(items), render_scan(items, meta, profil)


def to_df(items):
    rows = [[i["nama"], round(i["gram"]), *[round(i[k], 1) for k in KEYS]] for i in items]
    return pd.DataFrame(rows, columns=COLS)


def totals(items):
    return {k: sum(i[k] for i in items) for k in KEYS}


def render_scan(items, meta, profil):
    t = totals(items)
    skor = int(meta.get("skor") or 0)
    warna = "#16a34a" if skor >= 7 else "#d97706" if skor >= 4 else "#dc2626"
    peringatan = ""
    if profil and profil.get("pantangan"):
        kata = [w.strip().lower() for w in profil["pantangan"].replace(";", ",").split(",") if w.strip()]
        teks = " ".join([i["nama"].lower() for i in items] + [a.lower() for a in meta.get("alergen", [])])
        kena = [w for w in kata if w in teks]
        if kena:
            peringatan = f"<div class='warn'>⚠️ Mengandung pantanganmu: <b>{html.escape(', '.join(kena))}</b></div>"
    pct = ""
    if profil:
        pct = f"<p>≈ <b>{t['kalori'] / profil['target']['kalori'] * 100:.0f}%</b> dari target kalori harianmu.</p>"
    alerg = ""
    if meta.get("alergen"):
        alerg = f"<p>Kemungkinan alergen: {html.escape(', '.join(meta['alergen']))}</p>"
    return f"""{peringatan}<div class="stats">
<div class="stat big"><span>Total kalori</span><b>{t['kalori']:.0f}</b><small>kkal</small></div>
<div class="stat"><span>Protein</span><b>{t['protein_g']:.1f} g</b></div>
<div class="stat"><span>Karbo</span><b>{t['karbo_g']:.1f} g</b></div>
<div class="stat"><span>Lemak</span><b>{t['lemak_g']:.1f} g</b></div>
<div class="stat"><span>Serat</span><b>{t['serat_g']:.1f} g</b></div>
<div class="stat"><span>Gula</span><b>{t['gula_g']:.1f} g</b></div>
<div class="stat"><span>Natrium</span><b>{t['natrium_mg']:.0f} mg</b></div>
<div class="stat"><span>Skor gizi</span><b style="color:{warna}">{skor}/10</b></div></div>
{pct}{alerg}<p><i>{html.escape(meta.get('catatan', ''))}</i></p>
<p>💡 <b>Saran:</b> {html.escape(meta.get('saran', ''))}</p>
<p class="disclaimer">Estimasi AI dari foto/teks, bisa meleset. Koreksi berat porsi di tabel lalu klik “Hitung ulang”.</p>"""


def hitung_ulang(df, items, meta, profil):
    if not items:
        raise gr.Error("Scan dulu.")
    baru = []
    for idx, it in enumerate(items):
        try:
            g = float(df.iloc[idx, 1])
        except Exception:  # noqa: BLE001
            g = it["gram"]
        r = g / it["gram"] if it["gram"] else 1
        n = dict(it)
        n["nama"] = str(df.iloc[idx, 0])
        n["gram"] = g
        for k in KEYS:
            n[k] = it[k] * r
        baru.append(n)
    return baru, to_df(baru), render_scan(baru, meta, profil)


def simpan(items, meta, waktu, log):
    if not items:
        raise gr.Error("Belum ada hasil scan.")
    log = list(log or [])
    for it in items:
        log.append(dict(tanggal=str(date.today()), waktu=waktu, makanan=it["nama"], gram=round(it["gram"]),
                        **{k: round(it[k], 1) for k in KEYS}))
    return log, f"✅ {len(items)} item ditambahkan ke **{waktu}**. Lihat di tab *Hari Ini*."


# ------------------------------------------------------------------ DASHBOARD HARIAN
def hari_ini(log):
    return [r for r in (log or []) if r.get("tanggal") == str(date.today())]


def bar(label, nilai, target, satuan, batas=False):
    p = min(nilai / target * 100, 100) if target else 0
    over = target and nilai > target
    warna = "#dc2626" if over and batas else "#16a34a"
    return (f"<div class='pbar'><span>{label}</span><div class='track'><div class='fill' style='width:{p:.0f}%;"
            f"background:{warna}'></div></div><em>{nilai:.0f} / {target:.0f} {satuan}</em></div>")


def dashboard(profil, log):
    rows = hari_ini(log)
    if not rows:
        return "<p>Belum ada catatan makan hari ini. Scan makanan dulu di tab Scan.</p>", pd.DataFrame()
    s = {k: sum(r[k] for r in rows) for k in KEYS}
    df = pd.DataFrame(rows).drop(columns=["tanggal"])
    if profil:
        t = profil["target"]
        out = (bar("Kalori", s["kalori"], t["kalori"], "kkal", True) + bar("Protein", s["protein_g"], t["protein"], "g")
               + bar("Karbo", s["karbo_g"], t["karbo"], "g", True) + bar("Lemak", s["lemak_g"], t["lemak"], "g", True)
               + bar("Serat", s["serat_g"], 25, "g") + bar("Gula", s["gula_g"], 50, "g", True)
               + bar("Natrium", s["natrium_mg"], 2000, "mg", True))
        sisa = t["kalori"] - s["kalori"]
        out += f"<p><b>{'Sisa' if sisa >= 0 else 'Lebih'} {abs(sisa):.0f} kkal</b> dari target hari ini.</p>"
    else:
        out = f"<p>Total hari ini: <b>{s['kalori']:.0f} kkal</b>, protein {s['protein_g']:.0f} g. Hitung target di tab Profil untuk melihat progres.</p>"
    return out, df


def evaluasi(profil, log, key):
    rows = hari_ini(log)
    if not rows:
        raise gr.Error("Belum ada catatan makan hari ini.")
    s = {k: round(sum(r[k] for r in rows)) for k in KEYS}
    t = profil["target"] if profil else None
    p = (f"Asupan hari ini: {json.dumps(s)}. Makanan: {', '.join(r['makanan'] for r in rows)}. "
         + (f"Target: {t['kalori']} kkal, protein {t['protein']} g, karbo {t['karbo']} g, lemak {t['lemak']} g. "
            f"Tujuan: {profil['tujuan']}. " if t else "")
         + "Evaluasi singkat (kekurangan/kelebihan) dan beri 3 saran konkret untuk makan malam/besok.")
    yield from stream(p, temperature=0.4, key=key)


def unduh(log):
    if not log:
        raise gr.Error("Riwayat masih kosong.")
    path = "riwayat_makan.csv"
    pd.DataFrame(log).to_csv(path, index=False)
    return path


# ------------------------------------------------------------------ COACH
def chat_fn(message, history, profil, log, key):
    ctx = ""
    if profil:
        t = profil["target"]
        ctx += f"[Pengguna: tujuan {profil['tujuan']}, target {t['kalori']} kkal, protein {t['protein']} g]\n"
    rows = hari_ini(log)
    if rows:
        ctx += f"[Sudah dimakan hari ini: {round(sum(r['kalori'] for r in rows))} kkal, protein {round(sum(r['protein_g'] for r in rows))} g]\n"
    riwayat = "\n".join(f"{h['role']}: {h['content']}" for h in (history or [])[-6:])
    yield from stream(f"{ctx}{riwayat}\nuser: {message}", key=key)


# ------------------------------------------------------------------ UI
CSS = """
.gradio-container {max-width: 1000px !important; margin: auto;}
.hero {background: linear-gradient(135deg,#0f766e,#22c55e); padding:26px 24px; border-radius:18px; margin-bottom:12px;}
.hero h1, .hero p {color:#fff !important; margin:0;} .hero p {margin-top:6px; opacity:.95;}
.stats {display:grid; grid-template-columns:repeat(auto-fit,minmax(125px,1fr)); gap:10px; margin:10px 0;}
.stat {background:var(--background-fill-secondary); border:1px solid var(--border-color-primary); border-radius:14px; padding:12px 14px;}
.stat span {font-size:.78rem; color:var(--body-text-color-subdued); display:block;}
.stat b {font-size:1.3rem;} .stat small {display:block; color:var(--body-text-color-subdued);}
.stat.big {background:linear-gradient(135deg,#0f766e,#22c55e); border:none;}
.stat.big span, .stat.big small, .stat.big b {color:#fff !important;} .stat.big b {font-size:2rem;}
.warn {background:#fef2f2; border:1px solid #fca5a5; color:#991b1b; padding:10px 14px; border-radius:12px; margin-bottom:8px;}
.pbar {display:grid; grid-template-columns:80px 1fr 150px; gap:10px; align-items:center; margin:8px 0;}
.track {background:var(--background-fill-secondary); border-radius:99px; height:12px; overflow:hidden; border:1px solid var(--border-color-primary);}
.fill {height:100%; border-radius:99px;} .pbar em {font-style:normal; font-size:.85rem; color:var(--body-text-color-subdued);}
.disclaimer {font-size:.8rem; color:var(--body-text-color-subdued); text-align:center; margin-top:8px;}
"""
theme = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate", font=[gr.themes.GoogleFont("Inter"), "sans-serif"])

with gr.Blocks(theme=theme, css=CSS, title="FitNutri AI") as demo:
    profil_s = gr.BrowserState(None, storage_key="fitnutri_profil")
    log_s = gr.BrowserState([], storage_key="fitnutri_log")
    items_s, meta_s = gr.State([]), gr.State({})
    key_s = gr.State("")

    gr.HTML('<div class="hero"><h1>🥗 FitNutri AI</h1>'
            '<p>Scan makanan, hitung nutrisi, dan pantau workout harian dengan Generative AI.</p></div>')

    with gr.Tabs():
        with gr.Tab("1 · Profil & Target"):
            with gr.Row():
                nama = gr.Textbox(label="Nama", value="Arya", scale=2)
                gender = gr.Radio(["Pria", "Wanita"], label="Gender", value="Pria", scale=2)
                usia = gr.Number(label="Usia", value=19, minimum=12, maximum=80, precision=0)
            with gr.Row():
                berat = gr.Number(label="Berat (kg)", value=65, minimum=30, maximum=250)
                tinggi = gr.Number(label="Tinggi (cm)", value=170, minimum=120, maximum=230)
                aktivitas = gr.Dropdown(list(AKTIVITAS), label="Aktivitas", value="1-3x/minggu")
            with gr.Row():
                tujuan = gr.Dropdown(TUJUAN, label="Tujuan", value=TUJUAN[1])
                level = gr.Dropdown(["Pemula", "Menengah", "Lanjut"], label="Level", value="Pemula")
                hari = gr.Slider(2, 6, step=1, value=4, label="Hari latihan / minggu")
            with gr.Row():
                alat = gr.Textbox(label="Alat tersedia", value="tanpa alat / dumbbell")
                pantangan = gr.Textbox(label="Pantangan / alergi (pisah koma)", placeholder="udang, kacang")
            b_hitung = gr.Button("🎯 Hitung Target", variant="primary", size="lg")
            kartu = gr.HTML()
            b_plan = gr.Button("✨ Buat Jadwal Workout & Menu dengan AI")
            with gr.Row():
                o_w = gr.Markdown()
                o_m = gr.Markdown()
            inputs = [nama, gender, usia, berat, tinggi, aktivitas, tujuan, level, hari, alat, pantangan]
            b_hitung.click(hitung_target, inputs, [profil_s, kartu])
            b_plan.click(rencana, [profil_s, key_s], [o_w, o_m])

        with gr.Tab("2 · Scan Nutrisi"):
            mode = gr.Radio(["Foto makanan", "Label kemasan", "Deskripsi teks"], value="Foto makanan", label="Mode scan")
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=320):
                    foto = gr.Image(label="Foto", type="pil", sources=["upload", "webcam"], height=300)
                    desk = gr.Textbox(label="Deskripsi (mode teks)", lines=2, placeholder="nasi goreng 1 piring + telur ceplok + es teh manis")
                    hint = gr.Textbox(label="Info tambahan (opsional)", placeholder="mis. nasi 1 centong, ayam dada tanpa kulit")
                    b_scan = gr.Button("🔍 Scan Nutrisi", variant="primary", size="lg")
                with gr.Column(scale=1, min_width=320):
                    o_scan = gr.HTML()
            gr.Markdown("**Koreksi porsi** (ubah kolom *Porsi (g)*, lalu hitung ulang):")
            tabel = gr.Dataframe(headers=COLS, interactive=True, wrap=True)
            with gr.Row():
                b_re = gr.Button("🔄 Hitung ulang", min_width=140)
                waktu = gr.Dropdown(["Sarapan", "Makan siang", "Makan malam", "Snack"], value="Makan siang", label="Waktu makan")
                b_save = gr.Button("➕ Simpan ke Hari Ini", variant="primary", min_width=180)
            msg = gr.Markdown()
            b_scan.click(scan, [foto, mode, desk, hint, profil_s, key_s], [items_s, meta_s, tabel, o_scan])
            b_re.click(hitung_ulang, [tabel, items_s, meta_s, profil_s], [items_s, tabel, o_scan])
            b_save.click(simpan, [items_s, meta_s, waktu, log_s], [log_s, msg])

        with gr.Tab("3 · Hari Ini"):
            b_dash = gr.Button("📊 Perbarui", variant="primary")
            o_dash = gr.HTML()
            t_log = gr.Dataframe(label="Catatan hari ini", interactive=False)
            with gr.Row():
                b_eval = gr.Button("🧠 Evaluasi AI", min_width=140)
                b_csv = gr.Button("⬇️ Unduh riwayat (CSV)")
            o_eval = gr.Markdown()
            f_csv = gr.File(label="File CSV", height=90)
            b_dash.click(dashboard, [profil_s, log_s], [o_dash, t_log])
            b_eval.click(evaluasi, [profil_s, log_s, key_s], o_eval)
            b_csv.click(unduh, log_s, f_csv)

        with gr.Tab("4 · Coach AI"):
            gr.ChatInterface(chat_fn, type="messages", additional_inputs=[profil_s, log_s, key_s],
                             examples=[["Pengganti ayam yang murah dan tinggi protein?"],
                                       ["Snack tinggi protein untuk anak kos?"],
                                       ["Cara mengurangi pegal setelah latihan?"]])

        with gr.Tab("⚙️ Pengaturan"):
            gr.Markdown("API key Gemini gratis: **aistudio.google.com/apikey**. Isi di sini (hanya untuk sesi ini), "
                        "atau set variabel `GEMINI_API_KEY` di server agar semua pengguna otomatis terlayani.")
            k = gr.Textbox(label="Gemini API key", type="password", placeholder="AIza...")
            k.change(lambda x: x, k, key_s)
            gr.Markdown(f"Model: `{MODEL}` (cadangan `{FALLBACK}`). Ganti lewat variabel `GEMINI_MODEL`.")

    gr.HTML('<div class="disclaimer">Saran AI bersifat umum, bukan pengganti dokter/ahli gizi. '
            'Foto dikirim ke Google Gemini untuk dianalisis.</div>')

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
