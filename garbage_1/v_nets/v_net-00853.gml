graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 853
  arrival_time 17619.12721878913
  lifetime 1025.0417598374681
  num_nodes 12
  type "path"
  node [
    id 0
    label "0"
    cpu 41
    gpu 40
    rom 38
  ]
  node [
    id 1
    label "1"
    cpu 11
    gpu 16
    rom 32
  ]
  node [
    id 2
    label "2"
    cpu 10
    gpu 5
    rom 36
  ]
  node [
    id 3
    label "3"
    cpu 21
    gpu 9
    rom 29
  ]
  node [
    id 4
    label "4"
    cpu 39
    gpu 2
    rom 17
  ]
  node [
    id 5
    label "5"
    cpu 8
    gpu 1
    rom 6
  ]
  node [
    id 6
    label "6"
    cpu 10
    gpu 5
    rom 45
  ]
  node [
    id 7
    label "7"
    cpu 36
    gpu 44
    rom 8
  ]
  node [
    id 8
    label "8"
    cpu 11
    gpu 39
    rom 10
  ]
  node [
    id 9
    label "9"
    cpu 39
    gpu 41
    rom 20
  ]
  node [
    id 10
    label "10"
    cpu 38
    gpu 19
    rom 27
  ]
  node [
    id 11
    label "11"
    cpu 14
    gpu 27
    rom 8
  ]
  edge [
    source 0
    target 1
    bw 48
  ]
  edge [
    source 1
    target 2
    bw 41
  ]
  edge [
    source 2
    target 3
    bw 50
  ]
  edge [
    source 3
    target 4
    bw 41
  ]
  edge [
    source 4
    target 5
    bw 11
  ]
  edge [
    source 5
    target 6
    bw 34
  ]
  edge [
    source 6
    target 7
    bw 6
  ]
  edge [
    source 7
    target 8
    bw 28
  ]
  edge [
    source 8
    target 9
    bw 4
  ]
  edge [
    source 9
    target 10
    bw 46
  ]
  edge [
    source 10
    target 11
    bw 13
  ]
]
