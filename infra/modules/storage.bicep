// Storage account (Standard LRS) with a blob container for raw documents.
// Access is via RBAC only — shared key access is disabled. The managed
// identity is granted Storage Blob Data Contributor.

@description('Name of the storage account (3-24 lowercase alphanumeric chars).')
param name string

@description('Azure region for the storage account.')
param location string

@description('Principal ID of the managed identity to grant access to.')
param principalId string

@description('Object ID of the GitHub Actions CI/CD service principal to grant access to.')
param githubActionsSpObjectId string

@description('Name of the blob container to create.')
param containerName string = 'raw-documents'

// Built-in role: Storage Blob Data Contributor
var blobDataContributorRoleId = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
// Built-in role: Storage Table Data Reader (read-only access to Table Storage)
var tableDataReaderRoleId = '76199698-9eea-407e-8d99-65c5e7c5d8b9'

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

// The API managed identity reads the `events` calendar table (api/calendar.py
// for the schedule_lookup agent tool). Read-only is sufficient; the table is
// written by the ingestion pipeline, not the API.
resource tableDataReaderAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, principalId, tableDataReaderRoleId)
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', tableDataReaderRoleId)
    principalId: principalId
    principalType: 'ServicePrincipal'
  }
}

// CI/CD: the GitHub Actions service principal also needs blob data access so the
// scheduled ingestion workflow can write raw documents to the container.
resource githubBlobDataContributorAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, githubActionsSpObjectId, blobDataContributorRoleId)
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', blobDataContributorRoleId)
    principalId: githubActionsSpObjectId
    principalType: 'ServicePrincipal'
  }
}

@description('Name of the storage account.')
output name string = storageAccount.name

@description('Primary blob endpoint.')
output endpoint string = storageAccount.properties.primaryEndpoints.blob

@description('Name of the raw documents container.')
output containerName string = rawDocumentsContainer.name
