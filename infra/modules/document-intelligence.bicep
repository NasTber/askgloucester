// Azure AI Document Intelligence (Form Recognizer, S0 tier) using RBAC.
// Local key auth is disabled and a custom subdomain is set so AAD tokens
// can be issued. The managed identity is granted Cognitive Services User.

@description('Name of the Document Intelligence account.')
param name string

@description('Azure region for the account.')
param location string

@description('Principal ID of the managed identity to grant access to.')
param principalId string

@description('Object ID of the GitHub Actions CI/CD service principal to grant access to.')
param githubActionsSpObjectId string

// Built-in role: Cognitive Services User
var cognitiveServicesUserRoleId = 'a97b65f3-24c7-4388-baec-2e87135dc908'

resource documentIntelligence 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: name
  location: location
  kind: 'FormRecognizer'
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

resource cognitiveServicesUserAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(documentIntelligence.id, principalId, cognitiveServicesUserRoleId)
  scope: documentIntelligence
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesUserRoleId)
    principalId: principalId
    principalType: 'ServicePrincipal'
  }
}

// CI/CD: the GitHub Actions service principal needs to call Document
// Intelligence so the scheduled ingestion workflow can analyze documents.
resource githubCognitiveServicesUserAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(documentIntelligence.id, githubActionsSpObjectId, cognitiveServicesUserRoleId)
  scope: documentIntelligence
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesUserRoleId)
    principalId: githubActionsSpObjectId
    principalType: 'ServicePrincipal'
  }
}

@description('Name of the Document Intelligence account.')
output name string = documentIntelligence.name

@description('Document Intelligence endpoint.')
output endpoint string = documentIntelligence.properties.endpoint
