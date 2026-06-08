// Azure OpenAI (S0 tier) using RBAC. Local key auth is disabled and a custom
// subdomain is set so AAD tokens can be issued. The managed identity is
// granted Cognitive Services OpenAI User. A text-embedding-3-small deployment
// backs the ingestion pipeline's vector embeddings.

@description('Name of the Azure OpenAI account.')
param name string

@description('Azure region for the account.')
param location string

@description('Principal ID of the managed identity to grant access to.')
param principalId string

@description('Name of the embedding model deployment.')
param embeddingDeploymentName string = 'text-embedding-3-small'

// Built-in role: Cognitive Services OpenAI User
var cognitiveServicesOpenAiUserRoleId = '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'

resource openAi 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: name
  location: location
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    customSubDomainName: name
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: true
  }
}

// text-embedding-3-small deployment used by ingestion/embedder.py.
resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: openAi
  name: embeddingDeploymentName
  sku: {
    name: 'Standard'
    capacity: 50
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-3-small'
      version: '1'
    }
  }
}

resource openAiUserAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(openAi.id, principalId, cognitiveServicesOpenAiUserRoleId)
  scope: openAi
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesOpenAiUserRoleId)
    principalId: principalId
    principalType: 'ServicePrincipal'
  }
}

@description('Name of the Azure OpenAI account.')
output name string = openAi.name

@description('Azure OpenAI endpoint.')
output endpoint string = openAi.properties.endpoint

@description('Name of the embedding model deployment.')
output embeddingDeploymentName string = embeddingDeployment.name
