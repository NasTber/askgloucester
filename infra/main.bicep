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

// Container image for the API. Required — must be supplied at deploy time
// (e.g. from GitHub Actions or the CLI) with the real
// `<loginServer>/askgloucester-api:<tag>` ref. No default is provided on
// purpose: a default would let a deploy that omits this parameter silently
// reset the Container App to a placeholder image.
@description('Container image reference for the API Container App. Required — pass the real <loginServer>/askgloucester-api:<tag> at deploy time (from GitHub Actions or the CLI). There is intentionally no default so an omitted value fails validation instead of reverting the app to a placeholder image.')
param containerImage string

// API runtime config that has no backing Bicep resource/output to read from
// (no chat-model deployment is provisioned, and the index name + API version
// are app-level constants). Defaults mirror .env; override per environment.
@description('Azure AI Search index name (AZURE_SEARCH_INDEX_NAME).')
param searchIndexName string = 'gloucester-documents'

@description('Azure OpenAI chat deployment name (AZURE_OPENAI_CHAT_DEPLOYMENT).')
param openAiChatDeployment string = 'gpt-4.1-mini'

@description('Azure OpenAI API version (AZURE_OPENAI_API_VERSION).')
param openAiApiVersion string = '2024-10-21'

// Object (principal) ID of the GitHub Actions service principal used by CI/CD
// (federated OIDC). Required — no default on purpose so an omitted value fails
// validation rather than silently skipping the CI/CD role assignments. The
// relevant modules grant this principal the build/deploy + data-plane roles the
// workflows need (AcrPush, Container App Contributor, Storage Blob Data
// Contributor, Cognitive Services User, Search Index Data Contributor, Search
// Service Contributor, Cognitive Services OpenAI User).
@description('Object ID of the GitHub Actions service principal for CI/CD RBAC. Required — no default.')
param githubActionsSpObjectId string

// Resource names derived from the environment name.
var identityName = 'id-askgloucester-${environmentName}'
var keyVaultName = 'kv-askgloucester-${environmentName}'
var storageAccountName = 'stakgloucester${environmentName}'
var searchName = 'srch-askgloucester-${environmentName}'
var documentIntelligenceName = 'docintel-askgloucester-${environmentName}'
// Live account is `aoai-…` (deployed with a uniqueString() subdomain), not the
// `openai-…` the template originally modelled. Match live; the module now
// references it as `existing`. See modules/openai.bicep.
var openAiName = 'aoai-askgloucester-${environmentName}'
// Container Apps plane. ACR names allow no hyphens, so it omits the dashes.
var containerAppName = 'ca-askgloucester-${environmentName}'
var containerAppsEnvName = 'cae-askgloucester-${environmentName}'
var logAnalyticsName = 'log-askgloucester-${environmentName}'
var containerRegistryName = 'acraskgloucester${environmentName}'

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
    githubActionsSpObjectId: githubActionsSpObjectId
  }
}

module search 'modules/search.bicep' = {
  name: 'search'
  params: {
    name: searchName
    location: location
    principalId: identity.outputs.principalId
    githubActionsSpObjectId: githubActionsSpObjectId
  }
}

module documentIntelligence 'modules/document-intelligence.bicep' = {
  name: 'documentIntelligence'
  params: {
    name: documentIntelligenceName
    location: location
    principalId: identity.outputs.principalId
    githubActionsSpObjectId: githubActionsSpObjectId
  }
}

// OpenAI is referenced as an existing account (no location — we don't own its
// lifecycle). This module manages its model deployments + RBAC only.
module openAi 'modules/openai.bicep' = {
  name: 'openai'
  params: {
    name: openAiName
    principalId: identity.outputs.principalId
    chatDeploymentName: openAiChatDeployment
    githubActionsSpObjectId: githubActionsSpObjectId
  }
}

// Hosting: Log Analytics + Container Apps env + ACR + the API Container App,
// running under the existing managed identity. Endpoints/names come straight
// from the data/AI module outputs so nothing is hardcoded; the index name,
// chat deployment and API version come from params (no backing resource).
module containerApp 'modules/containerapp.bicep' = {
  name: 'containerApp'
  params: {
    location: location
    containerAppName: containerAppName
    managedEnvironmentName: containerAppsEnvName
    logAnalyticsName: logAnalyticsName
    containerRegistryName: containerRegistryName
    identityResourceId: identity.outputs.id
    identityPrincipalId: identity.outputs.principalId
    identityClientId: identity.outputs.clientId
    githubActionsSpObjectId: githubActionsSpObjectId
    containerImage: containerImage
    keyVaultEndpoint: keyVault.outputs.endpoint
    storageAccountName: storage.outputs.name
    rawDocumentsContainer: storage.outputs.containerName
    documentIntelligenceEndpoint: documentIntelligence.outputs.endpoint
    searchEndpoint: search.outputs.endpoint
    searchIndexName: searchIndexName
    openAiEndpoint: openAi.outputs.endpoint
    openAiEmbeddingDeployment: openAi.outputs.embeddingDeploymentName
    // Consume the chat deployment name the openai module actually manages,
    // rather than threading the raw param through a second path.
    openAiChatDeployment: openAi.outputs.chatDeploymentName
    openAiApiVersion: openAiApiVersion
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

@description('Container App name.')
output containerAppName string = containerApp.outputs.name

@description('Public FQDN of the Container App.')
output containerAppFqdn string = containerApp.outputs.fqdn

@description('Container Registry name.')
output containerRegistryName string = containerApp.outputs.containerRegistryName

@description('Container Registry login server (image push/pull host).')
output containerRegistryLoginServer string = containerApp.outputs.containerRegistryLoginServer
