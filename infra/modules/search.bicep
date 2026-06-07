// Azure AI Search (Free tier) using RBAC for data and control plane.
// Local API key auth is disabled. The managed identity is granted
// Search Index Data Contributor and Search Service Contributor.

@description('Name of the Azure AI Search service.')
param name string

@description('Azure region for the search service.')
param location string

@description('Principal ID of the managed identity to grant access to.')
param principalId string

// Built-in roles
var searchIndexDataContributorRoleId = '8ebe5a00-799e-43f5-93ac-243d3dce84a7'
var searchServiceContributorRoleId = '7ca78c08-252a-4471-8644-bb5ff32d4ba0'

resource search 'Microsoft.Search/searchServices@2024-03-01-preview' = {
  name: name
  location: location
  sku: {
    name: 'free'
  }
  properties: {
    replicaCount: 1
    partitionCount: 1
    hostingMode: 'default'
    publicNetworkAccess: 'enabled'
    disableLocalAuth: true
  }
}

resource indexDataContributorAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(search.id, principalId, searchIndexDataContributorRoleId)
  scope: search
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', searchIndexDataContributorRoleId)
    principalId: principalId
    principalType: 'ServicePrincipal'
  }
}

resource serviceContributorAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(search.id, principalId, searchServiceContributorRoleId)
  scope: search
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', searchServiceContributorRoleId)
    principalId: principalId
    principalType: 'ServicePrincipal'
  }
}

@description('Name of the search service.')
output name string = search.name

@description('Search service endpoint.')
output endpoint string = 'https://${search.name}.search.windows.net'
