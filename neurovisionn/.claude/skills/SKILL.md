---
name: cost-estimate
description: Estimate development cost of a codebase based on lines of code and complexity
---

# Cost Estimate Command

You are a senior software engineering consultant tasked with estimating the development cost of the current codebase.

## Step 1: Analyze the Codebase

First, detect the project's languages, frameworks, and tech stack. Read the entire codebase to understand:
- Total lines of code by language
- Architectural complexity (frameworks, integrations, APIs)
- Advanced or specialized features
- Testing coverage
- Documentation quality

Use the Glob and Read tools to systematically review:
- All source files (detect languages from file extensions: `.ts`, `.py`, `.go`, `.rs`, `.java`, `.swift`, `.cpp`, `.rb`, `.cs`, `.php`, `.kt`, `.scala`, `.ex`, `.hs`, etc.)
- All configuration/build files (`package.json`, `Cargo.toml`, `go.mod`, `pyproject.toml`, `Gemfile`, `pom.xml`, `build.gradle`, `Makefile`, `CMakeLists.txt`, `docker-compose.yml`, etc.)
- All test files
- Infrastructure and deployment configs
- Documentation files

**Auto-detect the stack** by examining:
- Package managers and dependency files
- Framework-specific config files (e.g., `next.config.js`, `django settings`, `rails config`)
- Docker/container configurations
- CI/CD pipelines (`.github/workflows`, `.gitlab-ci.yml`, `Jenkinsfile`)
- Cloud/infrastructure configs (Terraform, Pulumi, CloudFormation, k8s manifests)

## Step 2: Calculate Development Hours

Based on industry standards for a **senior developer** (5+ years experience):

**Hourly Productivity Estimates by Code Category**:

| Category | Lines/Hour | Examples |
|----------|-----------|----------|
| Simple CRUD/boilerplate | 30-50 | REST endpoints, form handlers, basic UI |
| Standard business logic | 20-30 | Services, controllers, data processing |
| Complex algorithms/logic | 15-25 | Search, optimization, state machines |
| Frontend UI/UX | 20-35 | Components, layouts, responsive design |
| API integrations | 15-25 | Third-party SDKs, OAuth, webhooks |
| Database/ORM layer | 20-30 | Migrations, queries, schema design |
| Systems programming | 10-20 | OS-level, drivers, memory management, C/C++/Rust |
| GPU/shader programming | 10-20 | CUDA, Metal, OpenGL, Vulkan, compute shaders |
| Compiler/language tooling | 8-15 | Parsers, ASTs, code generation |
| Real-time/streaming | 10-20 | WebSocket, audio/video, event-driven |
| Security/crypto | 10-20 | Auth systems, encryption, certificate handling |
| ML/AI pipeline | 10-20 | Model training, data pipelines, inference |
| Infrastructure as Code | 15-25 | Terraform, k8s manifests, CI/CD |
| Embedded/firmware | 8-15 | Hardware interfaces, RTOS, bare-metal |
| Comprehensive tests | 25-40 | Unit, integration, e2e tests |

**Additional Time Factors**:
- Architecture & design: +15-20% of coding time
- Debugging & troubleshooting: +25-30% of coding time
- Code review & refactoring: +10-15% of coding time
- Documentation: +10-15% of coding time
- Integration & testing: +20-25% of coding time
- Learning curve (new frameworks/tools): +10-20% for specialized tech
- DevOps & deployment setup: +5-10% of coding time

**Calculate total hours** considering:
1. Base coding hours (lines of code / productivity rate, categorized by complexity)
2. Multipliers for complexity and overhead
3. Phases completed vs. remaining (if project uses phases)
4. Specialized knowledge required (identify domain-specific expertise)

## Step 3: Research Market Rates

Use WebSearch to find current hourly rates for:
- Senior developers in the project's primary language/stack
- Specialists for any advanced domains identified (ML, systems, embedded, etc.)
- Contractors vs. employees
- Geographic variations (US markets: SF Bay Area, NYC, Austin, Remote; or relevant regions)

Search queries to use (adapt to detected stack):
- "senior [primary language] developer hourly rate [current year]"
- "senior [framework] developer contractor rate [current year]"
- "senior software engineer hourly rate United States [current year]"
- "[specialist domain] developer freelance rate [current year]"

## Step 4: Calculate Organizational Overhead

Real companies don't have developers coding 40 hours/week. Account for typical organizational overhead to convert raw development hours into realistic calendar time.

**Weekly Time Allocation for Typical Company**:

| Activity | Hours/Week | Notes |
|----------|------------|-------|
| **Pure coding time** | 20-25 hrs | Actual focused development |
| Daily standups | 1.25 hrs | 15 min x 5 days |
| Weekly team sync | 1-2 hrs | All-hands, team meetings |
| 1:1s with manager | 0.5-1 hr | Weekly or biweekly |
| Sprint planning/retro | 1-2 hrs | Per week average |
| Code reviews (giving) | 2-3 hrs | Reviewing teammates' work |
| Slack/email/async | 3-5 hrs | Communication overhead |
| Context switching | 2-4 hrs | Interruptions, task switching |
| Ad-hoc meetings | 1-2 hrs | Unplanned discussions |
| Admin/HR/tooling | 1-2 hrs | Timesheets, tools, access requests |

**Coding Efficiency Factor**:
- **Startup (lean)**: 60-70% coding time (~24-28 hrs/week)
- **Growth company**: 50-60% coding time (~20-24 hrs/week)
- **Enterprise**: 40-50% coding time (~16-20 hrs/week)
- **Large bureaucracy**: 30-40% coding time (~12-16 hrs/week)

**Calendar Weeks Calculation**:
```
Calendar Weeks = Raw Dev Hours / (40 x Efficiency Factor)
```

## Step 5: Calculate Full Team Cost

Engineering doesn't ship products alone. Calculate the fully-loaded team cost including all supporting roles.

**Supporting Role Ratios** (expressed as ratio to engineering hours):

| Role | Ratio to Eng Hours | Typical Rate | Notes |
|------|-------------------|--------------|-------|
| Product Management | 0.25-0.40x | $125-200/hr | PRDs, roadmap, stakeholder mgmt |
| UX/UI Design | 0.20-0.35x | $100-175/hr | Wireframes, mockups, design systems |
| Engineering Management | 0.12-0.20x | $150-225/hr | 1:1s, hiring, performance, strategy |
| QA/Testing | 0.15-0.25x | $75-125/hr | Test plans, manual testing, automation |
| Project/Program Management | 0.08-0.15x | $100-150/hr | Schedules, dependencies, status |
| Technical Writing | 0.05-0.10x | $75-125/hr | User docs, API docs, internal docs |
| DevOps/Platform | 0.10-0.20x | $125-200/hr | CI/CD, infra, deployments |

**Team Composition by Company Stage**:

| Stage | PM | Design | EM | QA | PgM | Docs | DevOps |
|-------|-----|--------|-----|-----|------|------|--------|
| Solo/Founder | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| Lean Startup | 15% | 15% | 5% | 5% | 0% | 0% | 5% |
| Growth Company | 30% | 25% | 15% | 20% | 10% | 5% | 15% |
| Enterprise | 40% | 35% | 20% | 25% | 15% | 10% | 20% |

**Full Team Multiplier**:
- **Solo/Founder**: 1.0x (just engineering)
- **Lean Startup**: ~1.45x engineering cost
- **Growth Company**: ~2.2x engineering cost
- **Enterprise**: ~2.65x engineering cost

**Calculation**:
```
Full Team Cost = Engineering Cost x Team Multiplier
```

## Step 6: Generate Cost Estimate

Provide a comprehensive estimate in this format:

---

## [Project Name] - Development Cost Estimate

**Analysis Date**: [Current Date]
**Detected Stack**: [Languages, frameworks, and key technologies]

### Codebase Metrics

- **Total Lines of Code**: [number]
  - [Language 1]: [number] lines
  - [Language 2]: [number] lines
  - [Language N]: [number] lines
  - Tests: [number] lines
  - Configuration/IaC: [number] lines
  - Documentation: [number] lines

- **Complexity Factors**:
  - Core frameworks: [list detected frameworks]
  - Specialized domains: [e.g., real-time, ML, systems-level, GPU, etc.]
  - Third-party integrations: [list key integrations]
  - Infrastructure complexity: [e.g., microservices, k8s, multi-cloud]

### Development Time Estimate

**Base Development Hours**: [number] hours

Breakdown by code category:

| Category | Lines | Productivity Rate | Hours |
|----------|-------|-------------------|-------|
| [Category 1] | [X] | [X] lines/hr | [X] hrs |
| [Category 2] | [X] | [X] lines/hr | [X] hrs |
| ... | ... | ... | ... |
| **Subtotal** | **[X]** | | **[X] hrs** |

**Overhead Multipliers**:
- Architecture & Design: +[X]% ([hours] hours)
- Debugging & Troubleshooting: +[X]% ([hours] hours)
- Code Review & Refactoring: +[X]% ([hours] hours)
- Documentation: +[X]% ([hours] hours)
- Integration & Testing: +[X]% ([hours] hours)
- Learning Curve: +[X]% ([hours] hours)
- DevOps & Deployment: +[X]% ([hours] hours)

**Total Estimated Hours**: [number] hours

### Realistic Calendar Time (with Organizational Overhead)

| Company Type | Efficiency | Coding Hrs/Week | Calendar Weeks | Calendar Time |
|--------------|------------|-----------------|----------------|---------------|
| Solo/Startup (lean) | 65% | 26 hrs | [X] weeks | ~[X] months |
| Growth Company | 55% | 22 hrs | [X] weeks | ~[X] years |
| Enterprise | 45% | 18 hrs | [X] weeks | ~[X] years |
| Large Bureaucracy | 35% | 14 hrs | [X] weeks | ~[X] years |

### Market Rate Research

**Senior Developer Rates ([current year])**:
- Low end: $[X]/hour (remote, mid-level market)
- Average: $[X]/hour (standard US market)
- High end: $[X]/hour (SF Bay Area, NYC, specialized)

**Recommended Rate for This Project**: $[X]/hour

*Rationale*: [Explain why — based on detected stack complexity, specialized domains, and market demand for those skills]

### Total Cost Estimate

| Scenario | Hourly Rate | Total Hours | **Total Cost** |
|----------|-------------|-------------|----------------|
| Low-end | $[X] | [hours] | **$[X,XXX]** |
| Average | $[X] | [hours] | **$[X,XXX]** |
| High-end | $[X] | [hours] | **$[X,XXX]** |

**Recommended Estimate (Engineering Only)**: **$[X,XXX] - $[X,XXX]**

### Full Team Cost (All Roles)

| Company Stage | Team Multiplier | Engineering Cost | **Full Team Cost** |
|---------------|-----------------|------------------|-------------------|
| Solo/Founder | 1.0x | $[X] | **$[X]** |
| Lean Startup | 1.45x | $[X] | **$[X]** |
| Growth Company | 2.2x | $[X] | **$[X]** |
| Enterprise | 2.65x | $[X] | **$[X]** |

**Role Breakdown (Growth Company Example)**:

| Role | Hours | Rate | Cost |
|------|-------|------|------|
| Engineering | [X] hrs | $[X]/hr | $[X] |
| Product Management | [X] hrs | $[X]/hr | $[X] |
| UX/UI Design | [X] hrs | $[X]/hr | $[X] |
| Engineering Management | [X] hrs | $[X]/hr | $[X] |
| QA/Testing | [X] hrs | $[X]/hr | $[X] |
| Project Management | [X] hrs | $[X]/hr | $[X] |
| Technical Writing | [X] hrs | $[X]/hr | $[X] |
| DevOps/Platform | [X] hrs | $[X]/hr | $[X] |
| **TOTAL** | **[X] hrs** | | **$[X]** |

### Grand Total Summary

| Metric | Solo | Lean Startup | Growth Co | Enterprise |
|--------|------|--------------|-----------|------------|
| Calendar Time | [X] | [X] | [X] | [X] |
| Total Human Hours | [X] | [X] | [X] | [X] |
| **Total Cost** | **$[X]** | **$[X]** | **$[X]** | **$[X]** |

### Assumptions

1. Rates based on US market averages ([current year])
2. Full-time equivalent allocation for all roles
3. Includes complete implementation of all identified features
4. Does not include:
   - Marketing & sales
   - Legal & compliance
   - Office/equipment
   - Hosting/infrastructure costs
   - Ongoing maintenance post-launch

### Comparison: AI-Assisted Development

**Estimated time savings with Claude Code**: [X]%
**Effective hourly rate with AI assistance**: ~$[X]/hour equivalent productivity

## Step 7: Calculate Claude ROI - Value Per Claude Hour

This is the most important metric for understanding AI-assisted development efficiency. It answers: **"What did each hour of Claude's actual working time produce?"**

### 7a: Determine Actual Claude Clock Time

**Method 1: Git History (preferred)**

Run `git log --format="%ai" | sort` to get all commit timestamps. Then:
1. **First commit** = project start
2. **Last commit** = current state
3. **Total calendar days** = last - first
4. **Cluster commits into sessions**: group commits within 4-hour windows as one session
5. **Estimate session duration**: each session ~ 1-4 hours of active Claude work (use commit density as signal)

**Session Duration Heuristics**:
- 1-2 commits in a window -> ~1 hour session
- 3-5 commits in a window -> ~2 hour session
- 6-10 commits in a window -> ~3 hour session
- 10+ commits in a window -> ~4 hour session

**Method 2: File Modification Timestamps (no git)**

Use file modification timestamps from source files. Apply same session clustering logic.

**Method 3: Fallback Estimate**

If no reliable timestamps, estimate from lines of code:
- Assume Claude writes 200-500 lines of meaningful code per hour
- Claude active hours ~ Total LOC / 350

### 7b: Calculate Value per Claude Hour

```
Value per Claude Hour = Total Code Value (from Step 5) / Estimated Claude Active Hours
```

Calculate across scenarios:

| Code Value Scenario | Claude Hours (est.) | Value per Claude Hour |
|--------------------|--------------------|-----------------------|
| Engineering only (avg) | [X] hrs | **$[X,XXX]/hr** |
| Full team equivalent (Growth Co) | [X] hrs | **$[X,XXX]/hr** |
| Full team equivalent (Enterprise) | [X] hrs | **$[X,XXX]/hr** |

### 7c: Claude Efficiency vs. Human Developer

**Speed Multiplier**:
```
Speed Multiplier = Human Dev Hours / Claude Active Hours
```

**Cost Efficiency**:
```
Human Cost = Human Hours x $[avg rate]/hr
Claude Cost = Subscription ($20-200/month) + API costs (estimate from project size)
Savings = Human Cost - Claude Cost
ROI = Savings / Claude Cost
```

### 7d: Output Format

Add this section to the final report:

---

### Claude ROI Analysis

**Project Timeline**:
- First commit / project start: [date]
- Latest commit: [date]
- Total calendar time: [X] days ([X] weeks)

**Claude Active Hours Estimate**:
- Total sessions identified: [X] sessions
- Estimated active hours: [X] hours
- Method: [git clustering / file timestamps / LOC estimate]

**Value per Claude Hour**:

| Value Basis | Total Value | Claude Hours | $/Claude Hour |
|-------------|-------------|--------------|---------------|
| Engineering only | $[X] | [X] hrs | **$[X,XXX]/Claude hr** |
| Full team (Growth Co) | $[X] | [X] hrs | **$[X,XXX]/Claude hr** |

**Speed vs. Human Developer**:
- Estimated human hours for same work: [X] hours
- Claude active hours: [X] hours
- **Speed multiplier: [X]x** (Claude was [X]x faster)

**Cost Comparison**:
- Human developer cost: $[X] (at $[avg rate]/hr avg)
- Estimated Claude cost: $[X] (subscription + API)
- **Net savings: $[X]**
- **ROI: [X]x** (every $1 spent on Claude produced $[X] of value)

**The headline number**: *Claude worked for approximately [X] hours and produced the equivalent of $[X] in professional development value -- roughly **$[X,XXX] per Claude hour**.*

---

---

## Notes

- Present the estimate in a clear, professional format suitable for sharing with stakeholders.
- Include confidence intervals and key assumptions.
- Highlight areas of highest complexity that drive cost.
- Adapt all language-specific references to match the actual detected stack.
- If the project spans multiple languages/stacks, break down estimates per component.
