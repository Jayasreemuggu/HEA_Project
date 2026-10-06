const API_URL = "http://hea-ml-api-v2.eu-north-1.elasticbeanstalk.com";

export async function POST(request: Request) {
  try {
    const body = await request.json();

    const response = await fetch(`${API_URL}/predict-material`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
      cache: "no-store",
    });

    const data = await response.json();

    return Response.json(data, {
      status: response.status,
    });
  } catch (error) {
    return Response.json(
      { detail: "Unable to reach prediction backend." },
      { status: 502 }
    );
  }
}
