import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";

const config = JSON.parse(readFileSync("studio/src-tauri/tauri.conf.json", "utf8"));
const packages = config.bundle.linux.deb.depends;
if (
  !Array.isArray(packages) ||
  packages.length === 0 ||
  !packages.every(
    (packageName) => typeof packageName === "string" && /^[a-z0-9][a-z0-9+.-]*$/.test(packageName),
  )
) {
  throw new Error("Studio Debian runtime dependencies must be explicit package names.");
}

const result = spawnSync(
  "sudo",
  ["apt-get", "install", "-y", "-o", "Acquire::Retries=3", ...packages],
  { stdio: "inherit" },
);
if (result.error) throw result.error;
process.exit(result.status ?? 1);
