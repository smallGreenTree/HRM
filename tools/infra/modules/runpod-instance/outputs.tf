output "pod_id" {
  description = "ID of the created RunPod instance"
  value       = runpod_pod.gpu_instance.id
}

output "pod_name" {
  description = "Name of the pod"
  value       = runpod_pod.gpu_instance.name
}

output "gpu_type" {
  description = "GPU type allocated"
  value       = var.gpu_type_id
}

output "gpu_count" {
  description = "Number of GPUs allocated"
  value       = runpod_pod.gpu_instance.gpu_count
}

output "machine_id" {
  description = "Machine ID where pod is running"
  value       = try(runpod_pod.gpu_instance.machine_id, "")
}

output "pod_hostname" {
  description = "Hostname to access the pod"
  value       = try(runpod_pod.gpu_instance.machine_id, "not-available")
}

output "ssh_connection_string" {
  description = "SSH connection information"
  value       = "Check RunPod console for SSH access: https://www.runpod.io/console/pods"
  sensitive   = false
}
