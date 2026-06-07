// Key Vault with RBAC authorization. The managed identity is granted
// Key Vault Secrets User so it can read secrets without access policies.

@description('Name of the Key Vault.')
param name string

@description('Azure region for the Key Vault.')
param location string

@description('Principal ID of the managed identity to grant access to.')
param principalId string

// Built-in role: Key Vault Secrets User
var keyVaultSecretsUserRoleId = '4633458b-17de-408a-b874-0445c86b69e6'

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: name
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: tenant().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    publicNetworkAccess: 'Enabled'
  }
}

resource secretsUserAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, principalId, keyVaultSecretsUserRoleId)
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', keyVaultSecretsUserRoleId)
    principalId: principalId
    principalType: 'ServicePrincipal'
  }
}

@description('Name of the Key Vault.')
output name string = keyVault.name

@description('Key Vault URI endpoint.')
output endpoint string = keyVault.properties.vaultUri
