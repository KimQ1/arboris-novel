<template>
  <n-modal
    v-model:show="showModal"
    preset="card"
    :title="modalTitle"
    :style="{ width: '600px' }"
    :bordered="false"
    :segmented="{ content: 'soft', footer: 'soft' }"
  >
    <div class="space-y-4">
      <div>
        <label class="block text-sm font-medium text-gray-700 mb-2">
          {{ inputLabel }}
        </label>
        <n-input
          v-model:value="briefDescription"
          type="textarea"
          :placeholder="inputPlaceholder"
          :rows="3"
          :disabled="isGenerating"
        />
      </div>

      <div v-if="expandedContent" class="mt-4">
        <label class="block text-sm font-medium text-gray-700 mb-2">
          AI 生成的详细内容
        </label>
        <div class="bg-slate-50 border border-slate-200 rounded-lg p-4">
          <p class="text-sm text-slate-700 whitespace-pre-line leading-relaxed">
            {{ expandedContent }}
          </p>
        </div>
      </div>

      <div v-if="errorMessage" class="mt-4">
        <n-alert type="error" :title="errorMessage" />
      </div>
    </div>

    <template #footer>
      <div class="flex justify-end gap-3">
        <n-button @click="handleCancel" :disabled="isGenerating">
          取消
        </n-button>
        <n-button
          v-if="!expandedContent"
          type="primary"
          @click="handleGenerate"
          :loading="isGenerating"
          :disabled="!briefDescription.trim()"
        >
          生成
        </n-button>
        <template v-else>
          <n-button @click="handleRegenerate" :loading="isGenerating">
            重新生成
          </n-button>
          <n-button type="primary" @click="handleConfirm" :disabled="isGenerating">
            确认使用
          </n-button>
        </template>
      </div>
    </template>
  </n-modal>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { NModal, NInput, NButton, NAlert } from 'naive-ui'
import { NovelAPI } from '@/api/novel'
import { globalAlert } from '@/composables/useAlert'

// Props interface
interface Props {
  show: boolean
  contentType: 'faction' | 'quest' | 'location' | 'item'
  projectId: string
}

interface Emits {
  (e: 'update:show', value: boolean): void
  (e: 'confirm', data: { name: string; description: string }): void
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()

const showModal = computed({
  get: () => props.show,
  set: (value) => emit('update:show', value)
})

const briefDescription = ref('')
const expandedContent = ref('')
const isGenerating = ref(false)
const errorMessage = ref('')

const contentTypeLabels: Record<string, { title: string; label: string; placeholder: string }> = {
  faction: {
    title: 'AI 生成阵营',
    label: '阵营简述',
    placeholder: '例如：天剑宗，修仙界最强剑修门派'
  },
  quest: {
    title: 'AI 生成任务',
    label: '任务简述',
    placeholder: '例如：寻找失落的上古神器'
  },
  location: {
    title: 'AI 生成地点',
    label: '地点简述',
    placeholder: '例如：幽暗森林，充满危险的禁地'
  },
  item: {
    title: 'AI 生成物品',
    label: '物品简述',
    placeholder: '例如：破天剑，传说中的神兵'
  }
}

const modalTitle = computed(() => contentTypeLabels[props.contentType]?.title || 'AI 生成内容')
const inputLabel = computed(() => contentTypeLabels[props.contentType]?.label || '简短描述')
const inputPlaceholder = computed(() => contentTypeLabels[props.contentType]?.placeholder || '请输入简短描述...')

// 重置状态
watch(() => props.show, (newVal) => {
  if (newVal) {
    briefDescription.value = ''
    expandedContent.value = ''
    errorMessage.value = ''
  }
})

const handleGenerate = async () => {
  if (!briefDescription.value.trim()) {
    globalAlert.showError('请输入简短描述', '提示')
    return
  }

  isGenerating.value = true
  errorMessage.value = ''

  try {
    const data = await NovelAPI.expandContent(
      props.projectId,
      props.contentType,
      briefDescription.value.trim()
    )

    expandedContent.value = data.expanded_content
    globalAlert.showSuccess('生成成功！')
  } catch (error: any) {
    console.error('AI 生成失败:', error)
    errorMessage.value = error.message || '生成失败，请重试'
    globalAlert.showError(error.message || '生成失败，请重试', '生成失败')
  } finally {
    isGenerating.value = false
  }
}

const handleRegenerate = async () => {
  expandedContent.value = ''
  await handleGenerate()
}

const handleConfirm = () => {
  if (!expandedContent.value) {
    globalAlert.showError('请先生成内容', '提示')
    return
  }

  // 提取名称（从简短描述中提取第一个逗号或冒号前的部分）
  const name = briefDescription.value.split(/[，：,:]/).shift()?.trim() || '未命名'

  emit('confirm', {
    name,
    description: expandedContent.value
  })

  showModal.value = false
  globalAlert.showSuccess('已添加到蓝图')
}

const handleCancel = () => {
  showModal.value = false
}
</script>

<style scoped>
/* 可以添加自定义样式 */
</style>

