# Bitext (+ ABCD) dataset for the 8-category classifier

Categories: **Account, Payment, Technical, Refund, Delivery, Subscription, General, Other**
(label order is in `ml/labels_bitext8.json`).

## 1. Run it

```powershell
# from the project root
mkdir data\raw\bitext, data\raw\abcd
Invoke-WebRequest https://raw.githubusercontent.com/bitext/customer-support-llm-chatbot-training-dataset/main/data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv -OutFile data\raw\bitext\bitext_27k.csv
Invoke-WebRequest https://github.com/asappresearch/abcd/raw/master/data/abcd_v1.1.json.gz -OutFile data\raw\abcd\abcd_v1.1.json.gz

python -m ml.make_extra_technical_data
python -m ml.prepare_bitext --extra-technical data/raw/extra_technical.csv
python -m ml.train --data data/raw/tickets_bitext.csv --labels ml/labels_bitext8.json
```

Output: `data/raw/tickets_bitext.csv` (columns `subject, message, category, source, intent, flags`,
which is what `ml/train.py` reads) and `data/processed/bitext_prep_report.json`.
The build is deterministic (seed 42). Optional: `--per-class`, `--abcd-per-class`,
`--extra-technical my_technical.csv` (a CSV with a `message` column). The included generator creates 400
deterministic synthetic Technical starter messages; replace or supplement them with real anonymized tickets.

## 2. Result of this build

| Category | Bitext | ABCD | Extra Technical | Total |
|---|---:|---:|---:|---:|
| Account | 1500 | 500 | 0 | 2000 |
| Delivery | 1500 | 500 | 0 | 2000 |
| General | 1500 | 498 | 0 | 1998 |
| Payment | 1500 | 426 | 0 | 1926 |
| Refund | 1500 | 451 | 0 | 1951 |
| Other | 1209 | 243 | 0 | 1452 |
| Subscription | 1261 | 156 | 0 | 1417 |
| Technical | 0 | 718 | 400 | 1118 |
| **All** | 9970 | 3492 | **400** | **13862** |

## 3. Bitext intent -> category (all 27 intents mapped, unknown intents abort the run)

| Category | Bitext intents |
|---|---|
| Account | create_account, delete_account, edit_account, switch_account, recover_password, registration_problems |
| Payment | payment_issue, check_payment_methods, check_invoice, get_invoice |
| Refund | get_refund, track_refund, check_refund_policy |
| Delivery | track_order, delivery_options, delivery_period, change_shipping_address, set_up_shipping_address |
| Subscription | newsletter_subscription, check_cancellation_fee |
| General | contact_customer_service, contact_human_agent, review, complaint |
| Other | place_order, change_order, cancel_order |
| Technical | none (Bitext has no technical intents) |

### Edge decisions (change them in `INTENT_TO_CATEGORY`, one line each)

* **cancel_order -> Other.** The examples cancel an existing order, matching ABCD's `order_issue/manage_cancel`.
   Subscription remains for newsletter and cancellation-fee requests.
* **complaint -> General**, alongside `review` and `contact_*`, because these Bitext examples concern feedback
   or dissatisfaction with the service rather than a specific order action.
* **place_order, change_order -> Other.** There is no dedicated Orders category.
* **review -> General** (feedback), **contact_* -> General** (how do I reach you).
* **payment_issue/invoices -> Payment**, not Technical: a failed or wrong payment is a payment problem.

## 4. ABCD subflow -> category

ABCD adds a Technical class (Bitext has none) and, in the other categories, a second writing style, so
"long, natural sentence" is not a shortcut for Technical. Only subflows whose meaning was checked on sample
conversations are used.

| Category | ABCD flow / subflow |
|---|---|
| Technical | troubleshoot_site / search_results, shopping_cart, slow_speed |
| Payment | troubleshoot_site / credit_card (card rejected), subscription_inquiry / manage_dispute_bill |
| Account | account_access / recover_password, recover_username, reset_2fa; manage_account / manage_change_name, manage_change_phone |
| Refund | product_defect / refund_initiate, refund_status, refund_update |
| Delivery | shipping_issue / status, missing, manage; order_issue / manage_upgrade, manage_downgrade (shipping speed/date) |
| Subscription | subscription_inquiry / manage_extension |
| General | storewide_query / policy, timing, pricing, membership questions |
| Other | order_issue / manage_cancel, manage_create (remove/add an item on an existing order) |

Left out as ambiguous: shipping_issue/cost, subscription_inquiry/manage_pay_bill, manage_account/manage_change_address
and status_*, product_defect/return_*, purchase_dispute, single_item_query.
Note: **credit_card is Payment, not Technical** (a rejected card is a payment fault).

## 5. Cleaning steps and what they removed

1. Unescape HTML, strip control characters, collapse whitespace.
2. Duplicate key = lower-case, digits masked, `{{placeholders}}` masked, punctuation removed, repeated letters collapsed
   (so "cancel order" and "cancel oorder" count as one). Removed: **3,420** Bitext rows (12.7%), plus **43** more
   after placeholders were filled (different templates became identical), **266** ABCD rows, **1** ABCD row that matched Bitext.
3. Conflict check: the same text under two categories is dropped (0 found).
4. `{{Order Number}}`, `{{Currency Symbol}}{{Refund Amount}}`, `{{Person Name}}`, `{{Account Type}}`, `{{Account Category}}`,
   `{{Delivery City}}`, `{{Delivery Country}}`, `{{Invoice Number}}` are replaced by realistic values (all 9 Bitext
   placeholders handled, none dropped). No `{{ }}` is left in any row (checked).
5. Length limits match the app: at least 10 characters (`MIN_MESSAGE_LEN`), at most 60 words.
6. ABCD: the first customer message that states the problem and contains the subflow cue is used; greetings and
   "my name is ..." are dropped; vague openers such as "Hello" or "I have a question" are skipped, so text and label agree.
7. Per-category cap 1500 (Bitext) and 500 (ABCD), sampled evenly across intents. After moving `complaint` to
   General, 529 extra Bitext candidates were trimmed to preserve the cap. The 400 generated Technical rows are
   retained and marked `source=extra`.
8. Final checks (the run aborts if any fails): only the 8 labels, every category >= 100 rows, no placeholders, no duplicates.

Kept on purpose: typos and colloquial wording (real tickets have them), and 573 rows with mild/strong language
(Bitext "offensive" flag), because angry customers are realistic.

## 6. Sanity checks (TF-IDF + logistic regression baseline, NOT the DistilBERT model)

* On a held-out set of ABCD conversations the builder never saw (trained on ABCD train split only):
  macro F1 0.972, accuracy 0.978, and only 0.1% of non-Technical rows predicted as Technical.
  Before the style-balancing step, 28.5% of long natural-style tickets from non-Technical flows were predicted
  as Technical.
* The random Bitext test split scores about 0.99 macro F1 for the same baseline. Do not read that as real-world
  quality (see limitations).
* `ml/train.py` loads the CSV with the project's own `prepare_frame` and splits it 70/15/15 without errors.

## 7. Limitations (read before trusting metrics)

1. **Bitext is synthetic** (generated text, 27 intents). Metrics on it will look better than on real tickets.
2. **Near-duplicates remain.** Bitext produces many paraphrases of the same request; a random split puts paraphrases
   in both train and test. Exact and typo-level duplicates are removed, paraphrases are not.
3. **Technical is still mostly synthetic and narrow.** Its 718 ABCD rows cover site faults (search, cart, slow
   pages); 400 additional synthetic starters cover app crashes, uploads, errors, sync, notifications, browser, and
   media issues. Replace or supplement them with diverse, anonymized real tickets before production use. The
   earlier ABCD-only baseline got 5 of 10 probes right for app crashes, login freezes, and upload errors.
4. **ABCD is role-play** (fictional customers and one fictional shop), not real customer tickets.
5. **Subscription is thin and partly odd**: Bitext only has newsletter and cancellation-fee texts, and ABCD adds
   156 extension requests. There are no real plan/renewal tickets.
6. Technical (1118) and Subscription (1417) are the smallest classes. Watch per-class recall in `ml.evaluate`.
7. The app categories, response templates, priority rules, sample generator, and SQL checks use the same eight
   labels. Re-run `sql/schema.sql` on an existing database to migrate old `Billing`, `Cancellation`, and
   `General Query` values to `Payment`, `Subscription`, and `General` before deploying the 8-class model.

## 8. Priority policy

Urgent text rules (fraud, unauthorized activity, compromised accounts, stolen items, or money deducted) always
produce **High** priority. Without a high-priority match, **Payment, Refund, Technical, Account, and Delivery**
are **Medium**; **Subscription, General, and Other** use the **Low** default. Priority is rule-based and is not
predicted by the classifier.

## 9. Licences and credit

* Bitext Customer Support dataset: Hugging Face lists `cdla-sharing-1.0`; source
  github.com/bitext/customer-support-llm-chatbot-training-dataset.
* ABCD (Chen et al., NAACL 2021): MIT licence, github.com/asappresearch/abcd.
Check both licence pages yourself before publishing the dataset or a model trained on it.
