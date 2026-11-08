# Security Policy

## Reporting a vulnerability

Please don't open a public issue for security problems. Use GitHub's private vulnerability reporting on this repository (Security tab, "Report a vulnerability"). Include steps to reproduce and the affected version.

You should get a reply within a week.

## Supported versions

VoiceQA is pre-alpha. Only the latest commit on `main` gets fixes.

## Responsible use

VoiceQA places real phone calls. Only test voice agents you own or have permission to test. Don't use it to call numbers you aren't authorized to call, to flood a line with calls, or to record people without their consent. Call recording laws differ by country and state, so check the rules that apply to you.

## Handling credentials

SIP credentials belong in local config files or environment variables, never in scenario files you share or commit. Recordings and reports can contain personal data, so treat the `runs/` and `recordings/` folders as sensitive.
