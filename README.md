# Containerlab Lab: Automation & Telemetry for BGP Routing Policies

## Overview

This lab simulates an ISP network environment with two WAN links (`Provider1` and `Provider2`). It deploys an **Automation Framework** for updating BGP routing policies and a **Telemetry Stack** for real-time observability of link quality to the providers. The proposed solution enables automatic failover of BGP routing policies based on network quality metrics.

---

## Solution Description

### Automation Framework

| Component | Description |
|-----------|-------------|
| **NetBox Node** | Single Source of Truth for network infrastructure. Models BGP policies using custom fields. Provides RESTful API for external integration and Webhooks to trigger automation events. <br/>*Source: [NetBox Labs](https://netboxlabs.com/)* |
| **GitLab CI/CD** | Automated pipeline for BGP policy configuration changes. GitLab Runner executes configuration changes. <br/>*Source: [GitLab](https://gitlab.com/)* |
| **Nornir Node** | Multi-vendor, multi-platform automation task orchestration. GitLab Runner registered here for Huawei core router access. <br/>*Source: [Nornir Docs](https://nornir.readthedocs.io/en/latest/)* |

### Telemetry Stack

| Component | Description |
|-----------|-------------|
| **MTR (Matt's Traceroute)** | Link diagnostics for BGP peers — monitors latency, jitter, and packet loss. Integrated with BGP failover script. <br/>*Source: [MTR on GitHub](https://github.com/traviscross/mtr)* |
| **Elasticsearch** | Storage and automatic indexing of telemetry events sent by the BGP failover script. <br/>*Source: [Elasticsearch](https://www.elastic.co/elasticsearch)* |
| **Grafana** | Dynamic visualization of latency metrics to BGP peers and link change tracking. Uses Elasticsearch as datasource. <br/>*Source: [Grafana](https://grafana.com/)* |

---

## Lab Architecture

![Network Topology](./images/topology.png)

## BGP Failover Workflow

![Failover Workflow](./images/workflow.png)

---

## Automation Framework Configuration

### 1. NetBox Node

- Install the NetBox container following [NetBox Docker Plugin Guide](https://github.com/netbox-community/netbox-docker/wiki/Using-Netbox-Plugins)
- Install the BGP plugin: [netbox-bpg](https://github.com/netbox-community/netbox-bgp.git)
- Deploy NetBox node:
  ```bash
  docker compose up -d
