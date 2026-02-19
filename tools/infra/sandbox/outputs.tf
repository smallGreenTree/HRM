output "pod_id" {
  description = "RunPod instance ID"
  value       = module.runpod_pod.pod_id
}

output "pod_name" {
  description = "RunPod instance name"
  value       = module.runpod_pod.pod_name
}

output "gpu_info" {
  description = "GPU allocation information"
  value = {
    type  = module.runpod_pod.gpu_type
    count = module.runpod_pod.gpu_count
  }
}

output "ssh_connection" {
  description = "SSH connection string"
  value       = module.runpod_pod.ssh_connection_string
}

output "pod_hostname" {
  description = "Pod hostname"
  value       = module.runpod_pod.pod_hostname
}
