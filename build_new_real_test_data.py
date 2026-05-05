import argparse
import calendar
import re
import time
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.official_data import DISTRICT_ALIASES, fetch_epid_report_links


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "data" / "test_2025_2026_real.csv"
CACHE_DIR = BASE_DIR / "data" / "raw" / "wer_pdfs"
WEATHER_CACHE_DIR = BASE_DIR / "data" / "raw" / "open_meteo"

CSV_COLUMNS = [
    "District",
    "Number_of_Cases",
    "Week_Start_Date",
    "Month",
    "Year",
    "Week",
    "Week_End_Date",
    "Avg Max Temp (°C)",
    "Avg Min Temp (°C)",
    "Avg Apparent Max Temp (°C)",
    "Avg Apparent Min Temp (°C)",
    "Total Precipitation (mm)",
    "Total Rain (mm)",
    "Avg Wind Speed (km/h)",
    "Max Wind Gusts (km/h)",
    "Weather Code",
    "Avg Daylight Duration (hours)",
    "Avg Sunrise Time",
    "Avg Sunset Time",
]

DISTRICTS = {
    "Ampara": {"lat": 7.2906, "lng": 81.6724},
    "Anuradhapura": {"lat": 8.3350, "lng": 80.4050},
    "Badulla": {"lat": 6.9897, "lng": 81.0561},
    "Batticaloa": {"lat": 7.7172, "lng": 81.6924},
    "Colombo": {"lat": 6.9271, "lng": 79.8612},
    "Galle": {"lat": 6.0535, "lng": 80.2207},
    "Gampaha": {"lat": 7.0890, "lng": 80.0167},
    "Hambantota": {"lat": 6.1240, "lng": 81.1198},
    "Jaffna": {"lat": 9.6615, "lng": 80.0255},
    "Kalutara": {"lat": 6.5844, "lng": 79.9600},
    "Kandy": {"lat": 7.2906, "lng": 80.6337},
    "Kegalle": {"lat": 7.2530, "lng": 80.3550},
    "Kilinochchi": {"lat": 9.3947, "lng": 80.3930},
    "Kurunegala": {"lat": 7.4833, "lng": 80.3631},
    "Mannar": {"lat": 8.9800, "lng": 79.9029},
    "Matale": {"lat": 7.4688, "lng": 80.6233},
    "Matara": {"lat": 5.9490, "lng": 80.5540},
    "Monaragala": {"lat": 6.8730, "lng": 81.3483},
    "Mullaitivu": {"lat": 9.2683, "lng": 80.8122},
    "NuwaraEliya": {"lat": 6.9497, "lng": 80.7891},
    "Polonnaruwa": {"lat": 7.9408, "lng": 81.0111},
    "Puttalam": {"lat": 8.0314, "lng": 79.8278},
    "Ratnapura": {"lat": 6.6833, "lng": 80.4000},
    "Trincomalee": {"lat": 8.5711, "lng": 81.2348},
    "Vavuniya": {"lat": 8.7500, "lng": 80.4970},
}

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "weather_code",
    "daylight_duration",
    "sunrise",
    "sunset",
]


def format_date(value):
    return f"{value.month}/{value.day}/{value.year}"


def download_pdf(url):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1].split("?", 1)[0]
    path = CACHE_DIR / filename
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()

    response = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    path.write_bytes(response.content)
    return response.content


def extract_pdf_text(pdf_bytes):
    try:
        import fitz

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    except ImportError:
        from pypdf import PdfReader
        import io

        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)


def month_number(name):
    lookup = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
    lookup.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
    return lookup[name.lower()[:3] if len(name) > 3 else name.lower()]


def _date_range_to_dates(match):
    start_day = int(match.group(1))
    start_month_text = match.group(2)
    end_day = int(match.group(3))
    end_month = month_number(match.group(4))
    end_year = int(match.group(5))

    if start_month_text:
        start_month = month_number(start_month_text)
        start_year = end_year
        if start_month > end_month:
            start_year -= 1
    else:
        start_month = end_month
        start_year = end_year
        if start_day > end_day:
            start_month = end_month - 1
            if start_month == 0:
                start_month = 12
                start_year -= 1

    return date(start_year, start_month, start_day), date(end_year, end_month, end_day)


def parse_header_year(text):
    normalized = re.sub(r"\s+", " ", text)
    table_start = normalized.find("Table 1")
    header_text = normalized[:table_start] if table_start >= 0 else normalized[:3000]
    match = re.search(r"Vol\.\s*\d+\s+No\.\s*\d+.*?\b(20\d{2})\b", header_text, re.I)
    return int(match.group(1)) if match else None


def correct_period_years_from_header(start_date, end_date, header_year):
    if not header_year or end_date.year == header_year:
        return start_date, end_date

    # Some early-January WER PDFs carry the previous year in the Table 1 date
    # even though the report header is for the new epidemiological year.
    if end_date.month <= 2 and abs(header_year - end_date.year) == 1:
        corrected_end = end_date.replace(year=header_year)
        corrected_start_year = header_year - 1 if start_date.month > end_date.month else header_year
        corrected_start = start_date.replace(year=corrected_start_year)
        return corrected_start, corrected_end

    return start_date, end_date


def parse_report_period(text):
    normalized = re.sub(r"\s+", " ", text)
    table_start = normalized.find("Table 1")
    table_text = normalized[table_start:] if table_start >= 0 else normalized
    header_year = parse_header_year(text)

    week_range_pattern = (
        r"(\d{1,2})(?:st|nd|rd|th)?\s*"
        r"(?:(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s*)?"
        r"[–-]\s*"
        r"(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+"
        r"(20\d{2})"
    )

    table_week_match = re.search(
        week_range_pattern + r"\s*\(\d{1,2}(?:st|nd|rd|th)?\s+Week\)",
        table_text,
        re.I,
    )
    if table_week_match:
        start_date, end_date = _date_range_to_dates(table_week_match)
        return correct_period_years_from_header(start_date, end_date, header_year)

    match = re.search(week_range_pattern, table_text, re.I)
    if not match:
        raise ValueError("Could not parse report period")

    start_date, end_date = _date_range_to_dates(match)
    return correct_period_years_from_header(start_date, end_date, header_year)


def parse_report_week(text):
    normalized = re.sub(r"\s+", " ", text)
    table_week = re.search(r"\((\d{1,2})(?:st|nd|rd|th)?\s+Week\)", normalized, re.I)
    if table_week:
        return int(table_week.group(1))

    volume_week = re.search(r"Vol\.\s*\d+\s+No\.\s*(\d{1,2})", normalized, re.I)
    if volume_week:
        return int(volume_week.group(1))

    raise ValueError("Could not parse report week")


def parse_dengue_case_rows(text, source_url):
    report_week = parse_report_week(text)
    start_date, end_date = parse_report_period(text)
    table_start = text.find("RDHS")
    table_text = text[table_start:] if table_start >= 0 else text

    rows = []
    for district, aliases in DISTRICT_ALIASES.items():
        match = None
        for alias in aliases:
            pattern = rf"(?m)^\s*{re.escape(alias)}\s+(\d+)\s+(\d+)\b"
            match = re.search(pattern, table_text)
            if match:
                break

        if not match:
            continue

        rows.append(
            {
                "District": district,
                "Number_of_Cases": int(match.group(1)),
                "Week_Start_Date": start_date,
                "Month": start_date.month,
                "Year": end_date.year,
                "Week": report_week,
                "Week_End_Date": end_date,
                "source_url": source_url,
            }
        )

    return rows


def collect_case_rows(start_year, end_year, max_reports=None):
    limit = max_reports or 120
    links = fetch_epid_report_links(limit=limit)
    rows = []
    used_reports = 0

    for index, url in enumerate(links, start=1):
        try:
            pdf_bytes = download_pdf(url)
            text = extract_pdf_text(pdf_bytes)
            report_rows = parse_dengue_case_rows(text, url)
            if not report_rows:
                print(f"[WER {index}/{len(links)}] skipped, no district rows: {url}")
                continue

            report_year = report_rows[0]["Year"]
            if start_year <= report_year <= end_year:
                rows.extend(report_rows)
                used_reports += 1
                print(
                    f"[WER {index}/{len(links)}] week {report_rows[0]['Week']} "
                    f"{report_year}: {len(report_rows)} district rows",
                    flush=True,
                )
            elif report_year < start_year:
                break
        except Exception as error:
            print(f"[WER {index}/{len(links)}] failed: {url} ({error})", flush=True)

    if not rows:
        raise RuntimeError("No WER case rows were collected")

    print(f"Collected {len(rows)} district-week case rows from {used_reports} reports.", flush=True)
    return rows


def fetch_daily_weather(district, lat, lng, start_date, end_date):
    WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = WEATHER_CACHE_DIR / f"{district}_{start_date}_{end_date}.csv"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return pd.read_csv(cache_path)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Asia/Colombo",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    last_error = None
    for attempt in range(1, 6):
        response = requests.get(url, params=params, timeout=60)
        if response.status_code == 429:
            wait_seconds = 30 * attempt
            print(
                f"Open-Meteo rate limit for {district}; waiting {wait_seconds}s "
                f"(attempt {attempt}/5)",
                flush=True,
            )
            time.sleep(wait_seconds)
            continue

        try:
            response.raise_for_status()
            payload = response.json()
            if "daily" not in payload:
                raise ValueError(f"Open-Meteo response missing daily data: {payload}")
            daily_df = pd.DataFrame(payload["daily"])
            daily_df.to_csv(cache_path, index=False)
            return daily_df
        except Exception as error:
            last_error = error
            wait_seconds = 10 * attempt
            print(
                f"Open-Meteo fetch failed for {district}: {error}; "
                f"retrying in {wait_seconds}s",
                flush=True,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Open-Meteo fetch failed for {district}: {last_error}")


def minutes_after_midnight(value):
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    return dt.hour * 60 + dt.minute + dt.second / 60


def mode_weather_code(values):
    values = [int(v) for v in values if pd.notna(v)]
    if not values:
        return 0
    return Counter(values).most_common(1)[0][0]


def aggregate_weather(daily_df, start_date, end_date):
    weather = daily_df.copy()
    weather["date"] = pd.to_datetime(weather["time"]).dt.date
    mask = (weather["date"] >= start_date) & (weather["date"] <= end_date)
    week = weather.loc[mask].copy()
    if week.empty:
        raise ValueError(f"No weather rows for {start_date} to {end_date}")

    return {
        "Avg Max Temp (°C)": round(float(week["temperature_2m_max"].mean()), 4),
        "Avg Min Temp (°C)": round(float(week["temperature_2m_min"].mean()), 4),
        "Avg Apparent Max Temp (°C)": round(float(week["apparent_temperature_max"].mean()), 4),
        "Avg Apparent Min Temp (°C)": round(float(week["apparent_temperature_min"].mean()), 4),
        "Total Precipitation (mm)": round(float(week["precipitation_sum"].sum()), 4),
        "Total Rain (mm)": round(float(week["rain_sum"].sum()), 4),
        "Avg Wind Speed (km/h)": round(float(week["wind_speed_10m_max"].mean()), 4),
        "Max Wind Gusts (km/h)": round(float(week["wind_gusts_10m_max"].max()), 4),
        "Weather Code": mode_weather_code(week["weather_code"]),
        "Avg Daylight Duration (hours)": round(float(week["daylight_duration"].mean()) / 3600, 8),
        "Avg Sunrise Time": round(float(week["sunrise"].map(minutes_after_midnight).mean()), 2),
        "Avg Sunset Time": round(float(week["sunset"].map(minutes_after_midnight).mean()), 2),
    }


def attach_weather(case_rows):
    min_date = min(row["Week_Start_Date"] for row in case_rows)
    max_date = max(row["Week_End_Date"] for row in case_rows)
    daily_weather_by_district = {}

    for index, (district, coords) in enumerate(DISTRICTS.items(), start=1):
        print(f"[Weather {index}/{len(DISTRICTS)}] {district}: {min_date} to {max_date}", flush=True)
        daily_weather_by_district[district] = fetch_daily_weather(
            district, coords["lat"], coords["lng"], min_date, max_date
        )
        time.sleep(1.0)

    final_rows = []
    for row in case_rows:
        if row["District"] not in daily_weather_by_district:
            continue

        weather = aggregate_weather(
            daily_weather_by_district[row["District"]],
            row["Week_Start_Date"],
            row["Week_End_Date"],
        )
        output_row = {
            **row,
            "Week_Start_Date": format_date(row["Week_Start_Date"]),
            "Week_End_Date": format_date(row["Week_End_Date"]),
            **weather,
        }
        output_row.pop("source_url", None)
        final_rows.append(output_row)

    return final_rows


def main():
    parser = argparse.ArgumentParser(
        description="Build real newer test data from WER dengue reports and Open-Meteo weather."
    )
    parser.add_argument("--start-year", type=int, default=2025)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--max-reports",
        type=int,
        default=None,
        help="Optional cap for debugging; default reads available 2025-2026 WER reports.",
    )
    args = parser.parse_args()

    case_rows = collect_case_rows(args.start_year, args.end_year, args.max_reports)
    final_rows = attach_weather(case_rows)

    df = pd.DataFrame(final_rows, columns=CSV_COLUMNS)
    df = df.sort_values(["Year", "Week", "District"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Saved {len(df)} rows to {args.output}", flush=True)
    print(f"Years: {int(df['Year'].min())}-{int(df['Year'].max())}", flush=True)
    print(f"Weeks: {int(df['Week'].min())}-{int(df['Week'].max())}", flush=True)
    print(f"Districts: {df['District'].nunique()}", flush=True)


if __name__ == "__main__":
    main()
