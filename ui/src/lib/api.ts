import type { License } from "./types";

let base = "";
let token = "";

export function connectTo(port: number, bearer: string): void {
  base = `http://127.0.0.1:${port}`;
  token = bearer;
}

export function eventsUrl(): string {
  return `${base}/api/events?token=${encodeURIComponent(token)}`;
}

export class ApiError extends Error {
  readonly status: number;
  readonly license: License | null;
  readonly detail: string;

  constructor(status: number, message: string, license: License | null, detail: string) {
    super(message);
    this.status = status;
    this.license = license;
    this.detail = detail;
  }
}

interface ErrorBody {
  error?: string;
  detail?: string;
  license?: License;
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = { Authorization: `Bearer ${token}` };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(base + path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (response.ok) {
    const text = await response.text();
    return (text ? JSON.parse(text) : undefined) as T;
  }
  const parsed = (await response.json().catch(() => ({}))) as ErrorBody;
  const message = parsed.error ?? (parsed.license ? parsed.license.title : `Request failed (${response.status}).`);
  throw new ApiError(response.status, message, parsed.license ?? null, parsed.detail ?? "");
}

export const api = {
  get: <T>(path: string) => request<T>("GET", path),
  put: <T = void>(path: string, body: unknown) => request<T>("PUT", path, body),
  post: <T = void>(path: string, body?: unknown) => request<T>("POST", path, body ?? {}),
  del: <T = void>(path: string) => request<T>("DELETE", path),
};

export function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
