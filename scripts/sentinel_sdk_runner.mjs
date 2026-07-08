import vm from "node:vm";

const DEFAULT_BOOTSTRAP_URL = "https://sentinel.openai.com/backend-api/sentinel/sdk.js";
const DEFAULT_WAIT_MS = 5000;
const DEFAULT_UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36";

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value ?? {}));
}

function parseSdkUrl(bootstrapCode) {
  const match = String(bootstrapCode || "").match(/script\.src\s*=\s*['"]([^'"]+sdk\.js)['"]/);
  if (!match) {
    throw new Error("sentinel_sdk_bootstrap_parse_failed");
  }
  return match[1];
}

function parseSdkVersion(sdkUrl) {
  const match = String(sdkUrl || "").match(/\/sentinel\/([^/]+)\/sdk\.js(?:\?|$)/);
  return match ? match[1] : "";
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function loadSdkCode(bootstrapUrl) {
  const bootstrapResp = await fetch(bootstrapUrl);
  if (!bootstrapResp.ok) {
    throw new Error(`sentinel_sdk_bootstrap_http_${bootstrapResp.status}`);
  }
  const bootstrapCode = await bootstrapResp.text();
  const sdkUrl = parseSdkUrl(bootstrapCode);
  const sdkResp = await fetch(sdkUrl);
  if (!sdkResp.ok) {
    throw new Error(`sentinel_sdk_http_${sdkResp.status}`);
  }
  return {
    bootstrapUrl,
    sdkUrl,
    sdkVersion: parseSdkVersion(sdkUrl),
    sdkCode: await sdkResp.text(),
  };
}

function buildSandbox({ sdkUrl, deviceId, flow, requirements, userAgent }) {
  const sentinelOrigin = new URL(sdkUrl).origin;
  const topMessageListeners = [];
  const iframeCache = new Map();
  let cookieStore = `oai-did=${encodeURIComponent(deviceId)}`;

  function dispatchMessageFromIframe(source, payload) {
    for (const listener of topMessageListeners) {
      listener({
        source,
        origin: sentinelOrigin,
        data: payload,
      });
    }
  }

  function buildIframeWindow() {
    return {
      postMessage(message) {
        const type = String(message?.type || "");
        const requestId = String(message?.requestId || "");
        const currentFlow = String(message?.flow || flow);
        const proof = message?.p ?? "";
        let result;
        if (type === "init") {
          result = {
            cachedChatReq: cloneJson(requirements),
            cachedProof: proof,
          };
        } else if (type === "token") {
          const now = Date.now();
          const previous = iframeCache.get(currentFlow);
          const expired = !previous || now - previous.lastFetchTime > 54e4 || previous.cachedProof !== proof;
          const nextValue = expired
            ? {
                cachedChatReq: cloneJson(requirements),
                cachedProof: proof,
                lastFetchTime: now,
              }
            : previous;
          iframeCache.set(currentFlow, nextValue);
          result = {
            cachedChatReq: nextValue.cachedChatReq,
            cachedProof: nextValue.cachedProof,
          };
        } else {
          result = null;
        }
        queueMicrotask(() => {
          dispatchMessageFromIframe(this, {
            type: "response",
            requestId,
            result,
          });
        });
      },
    };
  }

  function createNode(tagName) {
    const listeners = new Map();
    const node = {
      tagName: String(tagName || "").toUpperCase(),
      style: {},
      src: "",
      async: false,
      defer: false,
      nonce: "",
      contentWindow: null,
      addEventListener(type, handler) {
        const items = listeners.get(type) || [];
        items.push(handler);
        listeners.set(type, items);
      },
      dispatch(type) {
        for (const handler of listeners.get(type) || []) {
          handler();
        }
      },
    };
    if (node.tagName === "IFRAME") {
      node.contentWindow = buildIframeWindow();
    }
    return node;
  }

  const currentScript = createNode("script");
  currentScript.src = sdkUrl;

  const documentElement = {
    getAttribute() {
      return "";
    },
  };

  const head = {
    appendChild() {
      return null;
    },
  };

  const body = {
    appendChild(node) {
      if (node?.tagName === "IFRAME") {
        setTimeout(() => node.dispatch("load"), 0);
      }
      return node;
    },
  };

  const document = {
    currentScript,
    scripts: [currentScript],
    documentElement,
    head,
    body,
    createElement(tagName) {
      return createNode(tagName);
    },
  };

  Object.defineProperty(document, "cookie", {
    get() {
      return cookieStore;
    },
    set(value) {
      cookieStore = String(value || "");
    },
  });

  const location = new URL("https://auth.openai.com/about-you");
  const navigator = {
    userAgent: userAgent || DEFAULT_UA,
    language: "en-US",
    languages: ["en-US", "en"],
    hardwareConcurrency: 8,
  };
  const screen = { width: 1920, height: 1080 };

  const sandbox = {
    console,
    setTimeout,
    clearTimeout,
    queueMicrotask,
    Promise,
    Date,
    Math,
    JSON,
    Array,
    Object,
    URL,
    URLSearchParams,
    fetch,
    Buffer,
    TextEncoder,
    TextDecoder,
    performance,
    crypto: globalThis.crypto,
    navigator,
    screen,
    location,
    document,
    window: null,
    self: null,
    globalThis: null,
    requestIdleCallback(callback) {
      return setTimeout(() => callback({ timeRemaining: () => 1, didTimeout: false }), 0);
    },
    cancelIdleCallback(handle) {
      clearTimeout(handle);
    },
    btoa(value) {
      return Buffer.from(String(value), "binary").toString("base64");
    },
    atob(value) {
      return Buffer.from(String(value), "base64").toString("binary");
    },
    addEventListener(type, handler) {
      if (type === "message") {
        topMessageListeners.push(handler);
      }
    },
  };
  sandbox.window = sandbox;
  sandbox.self = sandbox;
  sandbox.globalThis = sandbox;
  sandbox.top = sandbox;
  sandbox.parent = sandbox;
  document.defaultView = sandbox;
  return sandbox;
}

async function generateTokens(payload) {
  const flow = String(payload?.flow || "").trim();
  const deviceId = String(payload?.deviceId || "").trim();
  const requirements = payload?.requirements && typeof payload.requirements === "object" ? payload.requirements : null;
  if (!flow) {
    throw new Error("missing_flow");
  }
  if (!deviceId) {
    throw new Error("missing_device_id");
  }
  if (!requirements?.token) {
    throw new Error("missing_sentinel_requirements");
  }
  const waitMs = Math.max(0, Number(payload?.observerWaitMs || DEFAULT_WAIT_MS) || DEFAULT_WAIT_MS);
  const { sdkCode, sdkUrl, sdkVersion, bootstrapUrl } = await loadSdkCode(
    String(payload?.bootstrapUrl || DEFAULT_BOOTSTRAP_URL),
  );
  const sandbox = buildSandbox({
    sdkUrl,
    deviceId,
    flow,
    requirements,
    userAgent: String(payload?.userAgent || DEFAULT_UA),
  });
  const context = vm.createContext(sandbox);
  vm.runInContext(sdkCode, context, { filename: "sentinel-sdk.js", timeout: 15000 });
  const sdk = context.SentinelSDK || context.window?.SentinelSDK;
  if (!sdk || typeof sdk.token !== "function" || typeof sdk.init !== "function") {
    throw new Error("sentinel_sdk_api_missing");
  }
  await sdk.init(flow);
  const sentinelToken = String((await sdk.token(flow)) || "");
  let soToken = "";
  if (requirements?.so?.required && typeof sdk.sessionObserverToken === "function") {
    await sleep(waitMs);
    soToken = String((await sdk.sessionObserverToken(flow)) || "");
  }
  return {
    bootstrap_url: bootstrapUrl,
    sdk_url: sdkUrl,
    sdk_version: sdkVersion,
    token: sentinelToken,
    token_length: sentinelToken.length,
    so_token: soToken,
    so_token_length: soToken.length,
    so_generated: Boolean(soToken),
  };
}

async function main() {
  const input = await new Promise((resolve, reject) => {
    let text = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      text += chunk;
    });
    process.stdin.on("end", () => resolve(text));
    process.stdin.on("error", reject);
  });
  const payload = JSON.parse(input || "{}");
  const result = await generateTokens(payload);
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

main().catch((error) => {
  const message = error instanceof Error ? `${error.message}\n${error.stack || ""}` : String(error);
  process.stderr.write(`${message}\n`);
  process.exit(1);
});
