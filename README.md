# Containerlab Lab: Automation and Telemetry Applied to BGP Routing Policies

## 📌 Overview

This lab simulates an Internet Service Provider (ISP) network environment with two WAN links (**Provider1** and **Provider2**). It implements:

- An **Automation Framework** for dynamic BGP routing policy updates  
- A **Telemetry Stack** for real-time observability of link quality  

The proposed solution enables **automatic BGP policy failover** based on network quality metrics.

---

## 🧩 Solution Architecture

### 🔧 Automation Framework

#### NetBox Node
- Single Source of Truth (SSoT) for network infrastructure  
- BGP policy modeling using custom fields  
- RESTful API for integration with external tools  
- Webhooks to trigger automation events  

🔗 Source: https://netboxlabs.com/

---

#### GitLab CI/CD
- Automated pipeline for BGP policy changes  
- GitLab Runner executes configuration changes  

🔗 Source: https://gitlab.com/

---

#### Nornir Node
- Multi-vendor, multi-platform automation orchestration  
- GitLab Runner registered for access to Huawei core router  

🔗 Source: https://nornir.readthedocs.io/en/latest/

---

### 📊 Telemetry Stack

#### MTR (Matt's Traceroute)
- BGP peer link diagnostics  
- Monitoring of latency, jitter, and packet loss  
- Integrated with BGP failover script  

🔗 Source: https://github.com/traviscross/mtr

---

#### Elasticsearch Node
- Storage and indexing of telemetry events  
- Data ingestion from BGP failover script  

🔗 Source: https://www.elastic.co/

---

#### Grafana
- Dynamic visualization of latency metrics  
- Link status monitoring and change tracking  
- Elasticsearch used as datasource  

🔗 Source: https://grafana.com/

---

## 🏗️ Lab Architecture

> 📷 *Insert network topology diagram here*

---

## 🔄 BGP Failover Workflow

> 📷 *Insert workflow diagram here*

---

## ⚙️ Automation Framework Configuration

### 1. NetBox Node

- Install NetBox container:  
  https://github.com/netbox-community/netbox-docker/wiki/Using-Netbox-Plugins  

- Install BGP plugin:  
  https://github.com/netbox-community/netbox-bgp.git  

- Deploy NetBox:
```bash
docker compose up -d
