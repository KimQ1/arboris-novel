<template>
  <TransitionRoot as="template" :show="show">
    <Dialog as="div" class="relative z-50" @close="$emit('close')">
      <TransitionChild as="template" enter="ease-out duration-300" enter-from="opacity-0" enter-to="opacity-100" leave="ease-in duration-200" leave-from="opacity-100" leave-to="opacity-0">
        <div class="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" />
      </TransitionChild>

      <div class="fixed inset-0 z-10 overflow-y-auto">
        <div class="flex min-h-full items-end justify-center p-4 text-center sm:items-center sm:p-0">
          <TransitionChild as="template" enter="ease-out duration-300" enter-from="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95" enter-to="opacity-100 translate-y-0 sm:scale-100" leave="ease-in duration-200" leave-from="opacity-100 translate-y-0 sm:scale-100" leave-to="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95">
            <DialogPanel class="relative transform overflow-hidden rounded-lg bg-white text-left shadow-xl transition-all sm:my-6 sm:w-full sm:max-w-lg">
              <div class="bg-white px-5 pt-6 pb-5 sm:px-6 sm:pt-6 sm:pb-5">
                <div class="flex flex-col gap-4 sm:flex-row sm:items-start">
                  <div class="mx-auto flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full bg-purple-100 sm:mx-0 sm:h-12 sm:w-12">
                    <svg class="h-6 w-6 text-purple-600" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M4 4a2 2 0 00-2 2v1h16V6a2 2 0 00-2-2H4zM18 9H2v5a2 2 0 002 2h12a2 2 0 002-2V9zM4 13a1 1 0 011-1h1a1 1 0 110 2H5a1 1 0 01-1-1zm5-1a1 1 0 100 2h1a1 1 0 100-2H9z"></path>
                    </svg>
                  </div>
                  <div class="text-center sm:flex-1 sm:text-left">
                    <DialogTitle as="h3" class="text-xl font-semibold leading-7 text-gray-900">重新生成后续大纲</DialogTitle>
                    <div class="mt-2">
                      <p class="text-base text-gray-500">
                        当前已有 <span class="font-semibold text-purple-600">{{ currentChapterCount }}</span> 章大纲。
                        请选择从哪一章开始重新生成，以及生成多少章。
                      </p>
                    </div>
                  </div>
                </div>
                <div class="mt-6 space-y-4">
                  <div>
                    <label for="startChapter" class="block text-base font-medium text-gray-700">起始章节</label>
                    <input 
                      type="number" 
                      name="startChapter" 
                      id="startChapter" 
                      v-model.number="startChapter" 
                      class="mt-2 block w-full rounded-xl border border-gray-200 bg-gray-50 px-4 py-3 text-lg shadow-sm focus:border-purple-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-purple-500" 
                      :min="1" 
                      :max="currentChapterCount + 1"
                    >
                    <p class="mt-1 text-sm text-gray-500">从第几章开始重新生成（包含该章）</p>
                  </div>
                  <div>
                    <label for="numChapters" class="block text-base font-medium text-gray-700">生成数量</label>
                    <input 
                      type="number" 
                      name="numChapters" 
                      id="numChapters" 
                      v-model.number="numChapters" 
                      class="mt-2 block w-full rounded-xl border border-gray-200 bg-gray-50 px-4 py-3 text-lg shadow-sm focus:border-purple-500 focus:bg-white focus:outline-none focus:ring-2 focus:ring-purple-500" 
                      min="1" 
                      max="20"
                    >
                    <div class="mt-3 flex flex-wrap justify-center gap-3">
                      <button 
                        v-for="count in [1, 2, 5, 10]" 
                        :key="count" 
                        @click="setNumChapters(count)"
                        :class="['px-5 py-2 text-base rounded-full transition-colors duration-150', numChapters === count ? 'bg-purple-600 text-white shadow-md' : 'bg-gray-200 text-gray-700 hover:bg-gray-300']"
                      >
                        {{ count }} 章
                      </button>
                    </div>
                  </div>
                  <div class="bg-purple-50 border border-purple-200 rounded-lg p-4">
                    <p class="text-sm text-purple-800">
                      <svg class="inline w-5 h-5 mr-1" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"></path>
                      </svg>
                      将从第 <span class="font-semibold">{{ startChapter }}</span> 章开始，生成 <span class="font-semibold">{{ numChapters }}</span> 章大纲
                      （第 {{ startChapter }} - {{ startChapter + numChapters - 1 }} 章）
                    </p>
                  </div>
                </div>
              </div>
              <div class="bg-gray-50 px-6 py-4 sm:flex sm:flex-row-reverse sm:px-8">
                <button 
                  type="button" 
                  class="inline-flex w-full justify-center rounded-lg border border-transparent bg-purple-600 px-5 py-3 text-base font-semibold text-white shadow-sm hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 sm:ml-3 sm:w-auto" 
                  @click="handleRegenerate"
                >
                  开始生成
                </button>
                <button 
                  type="button" 
                  class="mt-3 inline-flex w-full justify-center rounded-lg border border-gray-300 bg-white px-5 py-3 text-base font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 sm:mt-0 sm:ml-3 sm:w-auto" 
                  @click="$emit('close')"
                >
                  取消
                </button>
              </div>
            </DialogPanel>
          </TransitionChild>
        </div>
      </div>
    </Dialog>
  </TransitionRoot>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { Dialog, DialogPanel, DialogTitle, TransitionChild, TransitionRoot } from '@headlessui/vue'

interface Props {
  show: boolean
  currentChapterCount: number
}

const props = defineProps<Props>()
const emit = defineEmits(['close', 'regenerate'])

const startChapter = ref(1)
const numChapters = ref(5)

// 当当前章节数变化时，默认从下一章开始生成
watch(() => props.currentChapterCount, (count) => {
  if (count > 0) {
    startChapter.value = count + 1
  } else {
    startChapter.value = 1
  }
}, { immediate: true })

const setNumChapters = (count: number) => {
  numChapters.value = count
}

const handleRegenerate = () => {
  if (numChapters.value > 0 && startChapter.value > 0) {
    emit('regenerate', {
      startChapter: startChapter.value,
      numChapters: numChapters.value
    })
    emit('close')
  }
}
</script>

