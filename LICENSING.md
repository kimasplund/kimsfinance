# kimsfinance Licensing

kimsfinance uses a **dual licensing** model to support both open source and commercial use.

---

## 🆓 Free for Individuals & Open Source

kimsfinance is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

**You can use it for free if:**
- ✅ You're an individual using it for personal projects
- ✅ You're releasing your application as open source under AGPL-3.0 or compatible license
- ✅ You're doing academic/educational research
- ✅ You open-source your entire application (including network services)

**See:** [LICENSE](LICENSE) for full AGPL-3.0 terms

---

## 💼 Paid for Commercial Use

If you're using kimsfinance in a **proprietary application** or as a **network service** without open-sourcing your code, you need a **Commercial License**.

**You need a commercial license if:**
- ❌ You run kimsfinance as an API/web service (chart generation service)
- ❌ You use it in proprietary trading systems (HTF, hedge funds, prop trading)
- ❌ You embed it in a closed-source SaaS product
- ❌ You don't want to open-source your application

**See:** [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) for pricing and details

---

## ⚖️ Why AGPL-3.0?

The **Affero GPL** is specifically designed for network services. Unlike regular GPL, AGPL requires you to:

> **Open-source your code if users interact with it over a network**

This means:
- Running kimsfinance as an API = must open-source OR buy commercial license
- Using in a web app = must open-source OR buy commercial license
- Internal trading tools accessed over network = must open-source OR buy commercial license

**Bottom line:** If you're making money with kimsfinance and don't want to reveal your secret sauce, you need a commercial license.

---

## 🎯 Quick Decision Guide

```
┌─────────────────────────────────────┐
│  How are you using kimsfinance?     │
└─────────────────────────────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  Personal research │──────► FREE (AGPL-3.0)
        │  or education?     │
        └────────────────────┘
                 │ No
                 ▼
        ┌────────────────────┐
        │  Open source your  │──────► FREE (AGPL-3.0)
        │  entire app?       │
        └────────────────────┘
                 │ No
                 ▼
        ┌────────────────────┐
        │  Running as a      │──────► PAID (Commercial License)
        │  network service?  │        From $999/year
        └────────────────────┘
```

---

## 💰 Commercial License Tiers

| Tier | Price | Best For |
|------|-------|----------|
| **Startup** | $999/year | Early-stage companies, <$1M revenue |
| **Business** | $4,999/year | Growing companies, unlimited usage |
| **Enterprise** | Contact us | Hedge funds, HTF firms, large institutions |

**Includes:** No open-source requirement + priority support + updates

**Contact:** licensing@asplund.kim

---

## 📚 Examples

### ✅ **FREE - You can use AGPL-3.0:**

1. **Personal Trading Dashboard**
   ```python
   # You: Individual trader building a personal dashboard
   # Usage: Local Python script, charts for your own analysis
   # License: FREE (AGPL-3.0)
   ```

2. **Open Source Trading Bot**
   ```python
   # You: Developer building an open-source trading bot on GitHub
   # Usage: Bot generates charts, entire codebase is open source
   # License: FREE (AGPL-3.0) - your bot must also be AGPL-3.0
   ```

3. **Academic Research**
   ```python
   # You: PhD student researching market patterns
   # Usage: Generate charts for papers, thesis
   # License: FREE (AGPL-3.0) - cite in publications
   ```

---

### 💰 **PAID - You need a Commercial License:**

1. **Trading Platform API**
   ```python
   # You: Company offering chart generation API
   # Usage: customers.com/api/charts (network service)
   # License: COMMERCIAL ($4,999/year Business tier)
   # Why: AGPL requires open-sourcing network services
   ```

2. **Hedge Fund Internal Tools**
   ```python
   # You: Hedge fund with proprietary trading system
   # Usage: Internal tools accessed by traders via web interface
   # License: COMMERCIAL (Enterprise tier, contact for pricing)
   # Why: Network service + won't open-source strategies
   ```

3. **HTF Algo Trading**
   ```python
   # You: High-frequency trading firm
   # Usage: Chart generation for monitoring systems
   # License: COMMERCIAL (Enterprise tier, contact for pricing)
   # Why: Proprietary algorithms, competitive edge
   ```

4. **SaaS Product**
   ```python
   # You: SaaS company embedding charts in your product
   # Usage: app.example.com shows charts to customers
   # License: COMMERCIAL ($4,999/year Business tier)
   # Why: Network service + closed source product
   ```

---

## 🤝 License Compatibility

### ✅ Compatible with AGPL-3.0:
- AGPL-3.0 (same license)
- GPL-3.0 or later
- Any license allowing AGPL usage

### ❌ Incompatible with AGPL-3.0:
- MIT, BSD, Apache (for network services)
- Proprietary/closed source
- **→ Need commercial license for these**

---

## 🛡️ Compliance

### If using AGPL-3.0:
1. Include license notice in your application
2. Provide source code access to network users
3. License your application under AGPL-3.0 compatible license
4. Include "Powered by kimsfinance" attribution

### If using Commercial License:
1. Display license key during initialization (optional)
2. No source code disclosure required
3. Include attribution (appreciated but not required)
4. Receive license certificate from us

---

## ❓ FAQ

**Q: I'm a solo developer making $0 revenue. Do I need a commercial license?**
A: No! Use AGPL-3.0 for free.

**Q: Can I evaluate before buying a commercial license?**
A: Yes! Use AGPL-3.0 version for evaluation. Buy commercial license before production deployment.

**Q: What if I want to switch from AGPL to Commercial later?**
A: Easy! Just purchase a commercial license anytime.

**Q: Do you audit license compliance?**
A: We don't actively audit, but AGPL violations are easily detectable (network services). We prefer collaborative resolution.

**Q: Can I get a refund?**
A: Yes, 30-day money-back guarantee if you're not satisfied.

**Q: Is there a free tier for nonprofits?**
A: Yes! 50% discount for 501(c)(3) organizations. Contact us.

---

## 📞 Contact

**General inquiries:** hello@asplund.kim
**Commercial licensing:** licensing@asplund.kim
**GitHub Issues:** https://github.com/kimasplund/kimsfinance/issues
**Website:** https://asplund.kim

---

## 📄 Legal

- **Open Source License:** [LICENSE](LICENSE) (AGPL-3.0)
- **Commercial License:** [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)
- **Copyright:** © 2025 Kim Asplund. All rights reserved.

---

**TL;DR:**
- 🆓 **Free** for individuals and open source projects (AGPL-3.0)
- 💰 **Paid** for companies using it in proprietary network services (from $999/year)
- 🎯 **Goal:** Make hedge funds and HTF firms pay while keeping it free for the community

**Questions? Read [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) or email licensing@asplund.kim**
