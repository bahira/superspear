// Answering Pomerado's questions in a terminal.
//
// The MCP server never asks anything itself: it publishes questions on a job and the calling
// agent answers them. For the command line, this module is the stand-in for that agent, and
// it follows the answer shapes Pomerado validates: an option id for a choice, an array of
// ids for a multiple choice, { confirmed } for a confirm, a plain string for text and secrets.

import { createInterface } from "node:readline/promises";
import { emitKeypressEvents } from "node:readline";

function label(option) {
  return option.accountSpecific === true ? (option.maskedLabel ?? "••••") : option.label;
}

function listOptions(question) {
  return question.options.map((option) => `  ${option.id} — ${label(option)}`).join("\n");
}

function maskedLine(prompt) {
  return new Promise((resolve, reject) => {
    const stdin = process.stdin;
    const stdout = process.stderr;
    if (!stdin.isTTY) {
      reject(new Error("A secret needs a terminal; this input is not interactive."));
      return;
    }
    emitKeypressEvents(stdin);
    const previousRaw = stdin.isRaw;
    stdin.setRawMode(true);
    stdin.resume();
    stdout.write(prompt);
    let value = "";
    const cleanup = () => {
      stdin.off("keypress", onKey);
      stdin.setRawMode(previousRaw);
      stdin.pause();
    };
    const onKey = (text, key) => {
      if (key?.ctrl && (key.name === "c" || key.name === "d")) {
        cleanup();
        stdout.write("\n");
        reject(new Error("Cancelled."));
        return;
      }
      if (key?.name === "return" || key?.name === "enter") {
        cleanup();
        stdout.write("\n");
        resolve(value);
        return;
      }
      if (key?.name === "backspace") {
        value = value.slice(0, -1);
        return;
      }
      if (text !== undefined && !key?.ctrl && !key?.meta) value += text;
    };
    stdin.on("keypress", onKey);
  });
}

/** Ask one question. A masked answer never shares the terminal with readline. */
async function askOne(question) {
  const prompt = `\n${question.prompt}`;

  if (question.type === "secret") {
    return await maskedLine(`${prompt}\n> `);
  }

  if (question.type === "credential") {
    process.stderr.write(`\nSign in to ${question.siteOrigin} (nothing is saved).\n`);
    let username;
    if (question.fields === "username_password") {
      const rl = createInterface({ input: process.stdin, output: process.stderr });
      try {
        username = await rl.question("Username or email: ");
      } finally {
        rl.close();
      }
    }
    const password = await maskedLine("Password: ");
    return {
      ...(username === undefined || username.trim() === "" ? {} : { username: username.trim() }),
      password,
      saveLogin: false,
    };
  }

  const rl = createInterface({ input: process.stdin, output: process.stderr });
  try {
    switch (question.type) {
      case "text": {
        const answer = await rl.question(`${prompt}\n> `);
        if (answer.trim() === "") throw new Error(`"${question.id}" needs an answer.`);
        return answer;
      }
      case "confirm": {
        const answer = await rl.question(`${prompt} [y/N] `);
        const confirmed = /^y(es)?$/i.test(answer.trim());
        return {
          confirmed,
          ...(confirmed && question.followUp?.defaultText !== undefined
            ? { text: question.followUp.defaultText }
            : {}),
        };
      }
      case "choice": {
        const answer = await rl.question(`${prompt}\n${listOptions(question)}\n> `);
        const picked = question.options.find(
          (option) =>
            option.id === answer.trim() ||
            label(option).toLowerCase() === answer.trim().toLowerCase(),
        );
        if (picked !== undefined) return picked.id;
        if (question.allowOther === true && answer.trim() !== "") return { other: answer.trim() };
        throw new Error(`"${question.id}" needs one of the listed options.`);
      }
      case "multi_choice": {
        const answer = await rl.question(
          `${prompt}\n${listOptions(question)}\nPick ids, separated by commas\n> `,
        );
        const picked = answer
          .split(",")
          .map((part) => part.trim())
          .filter((part) => part !== "")
          .map((part) => {
            const option = question.options.find((candidate) => candidate.id === part);
            if (option === undefined) throw new Error(`"${part}" is not one of the listed options.`);
            return option.id;
          });
        const minimum = question.minSelections ?? 0;
        if (picked.length < minimum) throw new Error(`Pick at least ${minimum}.`);
        return picked;
      }
      default:
        throw new Error(`This terminal cannot answer a ${String(question.type)} question.`);
    }
  } finally {
    rl.close();
  }
}

/** Ask every question of one request in the terminal and return the answers object. */
export async function askInTerminal(request) {
  if (typeof request.notice === "string") process.stderr.write(`\n${request.notice}\n`);
  const answers = {};
  for (const question of request.questions ?? []) {
    answers[question.id] = await askOne(question);
  }
  return answers;
}
