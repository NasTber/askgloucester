// AskGloucester infrastructure orchestrator.
// Deploys a managed identity and the core data/AI services, wiring RBAC so
// the identity has the access it needs without any access keys.
//
// Scope: resource group (deploy with `az deployment group create`).

targetScope = 'resourceGroup'

@description('Azure region for all resources.')
param location string = 'eastus'

@description('Environment name used to suffix resource names (e.g. dev, prod).')
param environmentName string = 'dev'

// Resource names derived from the environment name.
var identityName = 'id-askgloucester-${environmentName}'
var keyVaultName = 'kv-askgloucester-${environmentName}'
var storageAccountName = 'stakgloucester${environmentName}'
var searchName = 'srch-askgloucester-${environmentName}'
var documentIntelligenceName = 'docintel-askgloucester-${environmentName}'
var openAiName = 'openai-askgloucester-${environmentName}'

module identity 'modules/identity.bicep' = {
  name: 'identity'
  params: {
    name: identityName
    location: location
  }
}

module keyVault 'modules/keyvault.bicep' = {
  name: 'keyvault'
  params: {
    name: keyVaultName
    location: location
    principalId: identity.outputs.principalId
  }
}

module storage 'modules/storage.bicep' = {
  name: 'storage'
  params: {
    name: storageAccountName
    location: location
    principalId: identity.outputs.principalId
  }
}

module search 'modules/search.bicep' = {
  name: 'search'
  params: {
    name: searchName
    location: location
    principalId: identity.outputs.principalId
  }
}

module documentIntelligence 'modules/document-intelligence.bicep' = {
  name: 'documentIntelligence'
  params: {
    name: documentIntelligenceName
    location: location
    principalId: identity.outputs.principalId
  }
}

module openAi 'modules/openai.bicep' = {
  name: 'openai'
  params: {
    name: openAiName
    location: location
    principalId: identity.outputs.principalId
  }
}

// --- Outputs ---

@description('Managed identity name.')
output identityName string = identity.outputs.name

@description('Managed identity resource ID.')
output identityId string = identity.outputs.id

@description('Managed identity client ID.')
output identityClientId string = identity.outputs.clientId

@description('Key Vault name.')
output keyVaultName string = keyVault.outputs.name

@description('Key Vault endpoint.')
output keyVaultEndpoint string = keyVault.outputs.endpoint

@description('Storage account name.')
output storageAccountName string = storage.outputs.name

@description('Storage account blob endpoint.')
output storageEndpoint string = storage.outputs.endpoint

@description('Raw documents container name.')
output rawDocumentsContainer string = storage.outputs.containerName

@description('Azure AI Search service name.')
output searchName string = search.outputs.name

@description('Azure AI Search endpoint.')
output searchEndpoint string = search.outputs.endpoint

@description('Document Intelligence account name.')
output documentIntelligenceName string = documentIntelligence.outputs.name

@description('Document Intelligence endpoint.')
output documentIntelligenceEndpoint string = documentIntelligence.outputs.endpoint

@description('Azure OpenAI account name.')
output openAiName string = openAi.outputs.name

@description('Azure OpenAI endpoint.')
output openAiEndpoint string = openAi.outputs.endpoint

@description('Azure OpenAI embedding deployment name.')
output openAiEmbeddingDeployment string = openAi.outputs.embeddingDeploymentName
