---
title: Incident runbook
department: engineering
visibility: internal
version: 2
status: active
effective: 2026-09-29
updated: 2026-09-29
---

# Incident runbook

## Severity levels

A Sev1 is checkout being down for all customers. A Sev2 is a feature broken for
some customers with a workaround. Anything else is a Sev3.

## First response

For a Sev1, page the on call engineer and open a bridge call within 10 minutes.
Post an update on the status page within 30 minutes, even if the cause is not
known yet.

## After the incident

Write a postmortem within 3 working days. It must name the trigger, the
detection gap and one concrete action with an owner. Postmortems are blameless.
