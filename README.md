# 🚀 Containerlab Laboratory: BGP Routing Policies Automation & Telemetry

> A simulated ISP network environment with automated BGP policy failover based on real-time link quality metrics.

---

## 📋 Summary

This laboratory simulates an Internet Service Provider (ISP) network environment with two WAN uplinks (**Provider1** and **Provider2**). It deploys:

- 🔧 An **Automation Framework** for dynamic BGP routing policy updates
- 📊 A **Telemetry Stack** for real-time observability of link quality with upstream providers

✅ **Key Benefit**: The proposed solution enables **automatic BGP routing policy failover** driven by network quality metrics (latency, jitter, packet loss).

---

## 🏗️ Proposed Solution Architecture

### 🤖 Automation Framework

| Component | Role | Key Features |
|-----------|------|-------------|
| **NetBox** | Single Source of Truth (SSoT) | • BGP policy modeling with custom fields<br>• RESTful API for external integrations<br>• Webhooks to trigger automation events<br>🔗 [netboxlabs.com](https://netboxlabs.com/) |
| **GitLab CI/CD** | Configuration Pipeline | • Automated pipeline for BGP policy changes<br>• GitLab Runner as configuration deployment executor<br>🔗 [gitlab.com](https://gitlab.com/) |
| **Nornir** | Automation Orchestrator | • Multi-vendor, multi-platform task orchestration<br>• Integrated with GitLab Runner for secure access to Huawei core routers<br>🔗 [nornir.readthedocs.io](https://nornir.readthedocs.io/en/latest/#) |

---

## 🔄 Architecture Diagram

```mermaid
graph LR
    A[NetBox: SSoT] -->|Webhooks| B(GitLab CI/CD)
    B -->|Trigger Pipeline| C[GitLab Runner]
    C -->|Execute via SSH| D[Nornir]
    D -->|Configure| E[Huawei Core Router]
    F[Telemetry Stack] -->|Metrics| A
    F -->|Alerts| B
```mermaid

## 📡 Telemetry Stack

| Component | Role | Key Features |
|-----------|------|-------------|
| **MTR (Matt's Traceroute)** | Link Quality Diagnostics | • Real-time monitoring of latency, jitter, and packet loss<br>• BGP peer path analysis<br>• Integrated with BGP Failover Script for automated decision-making<br>🔗 [github.com/traviscross/mtr](https://github.com/traviscross/mtr) |
| **Elasticsearch** | Telemetry Data Backend | • Automatic indexing and storage of telemetry events<br>• High-performance search and aggregation for time-series metrics<br>• Receives data from BGP Failover Script via HTTP/REST<br>🔗 [elastic.co/elasticsearch](https://www.elastic.co/es/elasticsearch) |
| **Grafana** | Visualization & Dashboards | • Dynamic panels for BGP peer latency metrics<br>• Real-time link change tracking and alerting<br>• Elasticsearch as native datasource for powerful queries<br>🔗 [grafana.com](https://grafana.com/) |

---

## 🗺️ Laboratory Architecture

![Network Topology Diagram](./images/topology-diagram.png)

> 📌 *Replace `./images/topology-diagram.png` with the actual path to your topology image. Recommended size: 1200x800px for optimal rendering.*

### 🔍 Architecture Highlights:
- **Dual WAN Uplinks**: Provider1 & Provider2 with independent BGP sessions
- **Core Router**: Huawei device managed via SSH through Nornir
- **Automation Plane**: NetBox → GitLab CI/CD → Nornir → Router
- **Telemetry Plane**: MTR → BGP Failover Script → Elasticsearch → Grafana

---

## 🔄 BGP Failover Workflow

![BGP Failover Workflow Diagram](./images/bgp-failover-workflow.png)

> 📌 *Replace `./images/bgp-failover-workflow.png` with the actual path to your workflow image.*

### Workflow Steps:
1. 📊 **Monitor**: MTR continuously probes BGP peers for latency, jitter, and packet loss
2. ⚙️ **Evaluate**: BGP Failover Script analyzes metrics against defined thresholds
3. 🎯 **Decision**: If degradation detected → trigger policy update via webhook
4. 🚀 **Execute**: GitLab CI/CD pipeline activates → Nornir pushes new config to Huawei router
5. ✅ **Validate**: Post-change verification + telemetry update in Elasticsearch
6. 📈 **Visualize**: Grafana dashboard reflects new active provider and link status

```mermaid
sequenceDiagram
    participant MTR
    participant Script as BGP Failover Script
    participant ES as Elasticsearch
    participant NetBox
    participant GitLab
    participant Nornir
    participant Router

    MTR->>Script: Send metrics (latency/jitter/loss)
    Script->>Script: Evaluate thresholds
    alt Degradation detected
        Script->>NetBox: Update policy via API
        NetBox->>GitLab: Trigger webhook
        GitLab->>Nornir: Run deployment pipeline
        Nornir->>Router: Apply new BGP config via SSH
        Router-->>Nornir: Confirmation
        Nornir->>ES: Log change event
        ES->>Grafana: Real-time dashboard update
    else Metrics OK
        Script->>ES: Log normal operation
    end
