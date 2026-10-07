import { fileURLToPath } from "node:url";
import { runProcess } from "./process.mjs";
import { IMAGE } from "./tools.mjs";

console.log("Checking Docker…");
try {
  await runProcess("docker", ["info", "--format", "{{.ServerVersion}}"], {
    timeout: 10_000,
  });
  console.log("Building the local training runtime…");
  const folder = fileURLToPath(new URL(".", import.meta.url));
  await runProcess("docker", ["build", "-t", IMAGE, folder], {
    timeout: 600_000,
    onLine: (line) => console.log(line),
  });
  console.log("Runtime ready. Run npm run build, then npm start.");
} catch (error) {
  console.error(
    "Start Docker Desktop or the Docker daemon, then run npm run setup again.\n" +
      error.message,
  );
  process.exitCode = 1;
}
