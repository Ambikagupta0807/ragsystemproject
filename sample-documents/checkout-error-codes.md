---
title: Checkout error codes
department: engineering
visibility: internal
version: 3
status: active
effective: 2026-09-18
updated: 2026-09-18
---

# Checkout error codes

## Payment errors

ERR-4010 means the payment gateway timed out. Ask the customer to retry after
five minutes. Any amount deducted is auto reversed within 48 hours.

ERR-4022 means the card was declined by the issuing bank. We cannot fix this,
the customer must contact their bank.

## Order errors

ERR-5021 means the inventory service did not respond while reserving stock. The
order is not created, and retrying usually works.

ERR-5090 means the address could not be validated for that pin code. Ask the
customer for a nearby serviceable pin code.
