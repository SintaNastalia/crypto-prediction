import requests

def get_latest_fgi():
    """
    Ambil nilai FGI terbaru dari API alternative.me
    """
    url = "https://api.alternative.me/fng/?limit=1&format=json"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Lempar error kalau status bukan 200
        data = response.json()

        # Ambil nilai FGI terbaru
        latest_fgi_value = int(data['data'][0]['value'])
        return latest_fgi_value
    except Exception as e:
        print(f"Gagal mengambil data FGI terbaru: {e}")
        return None
