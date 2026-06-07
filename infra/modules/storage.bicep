// Storage account (Standard LRS) with a blob container for raw documents.
// Access is via RBAC only — shared key access is disabled. The managed
// identity is granted Storage Blob Data Contributor.

@description('Name of the storage account (3-24 lowercase alphanumeric chars).')
param name string

@description('Azure region for the storage account.')
param location string

@description('Principal ID of the managed identity to grant access to.')
param principalId string

@description('Name of the blob container to create.')
param containerName string = 'raw-documents'

// Built-in role: Storage Blob Data Contributor
var blobDataContributorRoleId = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: name
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storageAccount
  name: 'default'
}

resource rawDocumentsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: containerName
  properties: {
    publicAccess: 'None'
  }
}

resource blobDataContributorAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, principalId, blobDataContributorRoleId)
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', blobDataContributorRoleId)
    principalId: principalId
    principalType: 'ServicePrincipal'
  }
}

@description('Name of the storage account.')
output name string = storageAccount.name

@description('Primary blob endpoint.')
output endpoint string = storageAccount.properties.primaryEndpoints.blob

@description('Name of the raw documents container.')
output containerName string = rawDocumentsContainer.name
