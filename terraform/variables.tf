variable "ENV" {
  type        = string
  description = "The prefix which should be used for all resources in this environment. Make it unique, like ksultanau."
  default = "development"
}

variable "LOCATION" {
  type        = string
  description = "The Azure Region in which all resources in this example should be created."
  default     = "westeurope"
}

variable "BDCC_REGION" {
  type        = string
  description = "The BDCC Region for billing."
  default     = "global"
}

variable "STORAGE_ACCOUNT_REPLICATION_TYPE" {
  type        = string
  description = "Storage Account replication type."
  default     = "LRS"
}

variable "IP_RULES" {
  type        = map(string)
  description = "Map of IP addresses permitted to access"
  default = {
    "epam-vpn-ru-0" = "185.44.13.36"
    "epam-vpn-eu-0" = "195.56.119.209"
    "epam-vpn-eu-1" = "195.56.119.212"
    "epam-vpn-eu-2" = "204.153.55.4"
    "epam-vpn-in-0" = "203.170.48.2"
    "epam-vpn-ua-0" = "85.223.209.18"
    "epam-vpn-us-0" = "174.128.60.160"
    "epam-vpn-us-1" = "174.128.60.162"
    "epam-vpn-by-0" = "213.184.231.20"
    "epam-vpn-by-1" = "86.57.255.94"
  }
}
