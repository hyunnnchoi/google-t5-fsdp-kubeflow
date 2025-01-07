#!/bin/bash

# Job 파일들이 있는 경로 설정
JOB_DIR="./"
# Job 파일들의 이름 패턴 (t{번호} 순서대로 실행)
JOB_FILES=($(ls ${JOB_DIR}*.yaml | sort -V))

# 모든 Job 파일 순서대로 실행
for job_file in "${JOB_FILES[@]}"; do
    # 현재 실행 중인 Job의 이름 출력
    echo "Running job: ${job_file}"

    # 현재 Job 실행
    kubectl create -f "${job_file}"

    # Job 번호 추출 (id{번호} 부분만 가져옴)
    job_number=$(basename "${job_file}" .yaml | grep -oE '^id[0-9]+')

    # 모든 관련 Pod들이 조건을 만족할 때까지 대기
    while true; do
        # 디버깅: 현재 Job 관련 모든 Pod 상태 출력
        echo "Debug: Checking all pods related to job ${job_number}"
        POD_OUTPUT=$(kubectl get pods --no-headers | grep "${job_number}")
        echo "${POD_OUTPUT}"

        # Controller Pod 확인
        CONTROLLER_EXISTS=$(echo "${POD_OUTPUT}" | grep "controller" | wc -l)
        echo "Debug: Controller pod exists: ${CONTROLLER_EXISTS}"

        if [ "${CONTROLLER_EXISTS}" -gt 0 ]; then
            # Controller Pod Completed 상태 확인
            CONTROLLER_COMPLETED=$(echo "${POD_OUTPUT}" | grep "controller" | grep "Completed" | wc -l)
            echo "Debug: Controller pods completed: ${CONTROLLER_COMPLETED} / ${CONTROLLER_EXISTS}"

            if [ "${CONTROLLER_COMPLETED}" -eq "${CONTROLLER_EXISTS}" ]; then
                echo "All controller pods for job ${job_number} are Completed."
                echo "Deleting job: ${job_file}"
                kubectl delete -f "${job_file}"
                break
            fi
        else
            # Worker Pod 상태 확인
            COMPLETED_WORKERS=$(echo "${POD_OUTPUT}" | grep "worker" | grep "Completed" | wc -l)
            TOTAL_WORKERS=$(echo "${POD_OUTPUT}" | grep "worker" | wc -l)

            echo "Debug: Completed workers: ${COMPLETED_WORKERS} / Total workers: ${TOTAL_WORKERS}"

            if [ "${COMPLETED_WORKERS}" -eq "${TOTAL_WORKERS}" ] && [ "${TOTAL_WORKERS}" -gt 0 ]; then
                echo "All worker pods for job ${job_number} are Completed."
                echo "Deleting job: ${job_file}"
                kubectl delete -f "${job_file}"
                break
            fi
        fi

        # 대기
        echo "Waiting for pods of job ${job_number} to complete..."
        sleep 5
    done
done
