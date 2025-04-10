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
  id 1228
  arrival_time 25407.93674325516
  lifetime 511.52584878056484
  num_nodes 15
  type "path"
  node [
    id 0
    label "0"
    cpu 35
    gpu 24
    rom 41
  ]
  node [
    id 1
    label "1"
    cpu 39
    gpu 20
    rom 16
  ]
  node [
    id 2
    label "2"
    cpu 14
    gpu 33
    rom 49
  ]
  node [
    id 3
    label "3"
    cpu 50
    gpu 20
    rom 39
  ]
  node [
    id 4
    label "4"
    cpu 41
    gpu 5
    rom 23
  ]
  node [
    id 5
    label "5"
    cpu 34
    gpu 41
    rom 7
  ]
  node [
    id 6
    label "6"
    cpu 14
    gpu 50
    rom 31
  ]
  node [
    id 7
    label "7"
    cpu 2
    gpu 7
    rom 30
  ]
  node [
    id 8
    label "8"
    cpu 45
    gpu 9
    rom 18
  ]
  node [
    id 9
    label "9"
    cpu 34
    gpu 15
    rom 34
  ]
  node [
    id 10
    label "10"
    cpu 24
    gpu 9
    rom 8
  ]
  node [
    id 11
    label "11"
    cpu 1
    gpu 17
    rom 17
  ]
  node [
    id 12
    label "12"
    cpu 11
    gpu 30
    rom 28
  ]
  node [
    id 13
    label "13"
    cpu 1
    gpu 17
    rom 18
  ]
  node [
    id 14
    label "14"
    cpu 43
    gpu 7
    rom 12
  ]
  edge [
    source 0
    target 1
    bw 34
  ]
  edge [
    source 1
    target 2
    bw 26
  ]
  edge [
    source 2
    target 3
    bw 27
  ]
  edge [
    source 3
    target 4
    bw 3
  ]
  edge [
    source 4
    target 5
    bw 29
  ]
  edge [
    source 5
    target 6
    bw 34
  ]
  edge [
    source 6
    target 7
    bw 48
  ]
  edge [
    source 7
    target 8
    bw 40
  ]
  edge [
    source 8
    target 9
    bw 43
  ]
  edge [
    source 9
    target 10
    bw 11
  ]
  edge [
    source 10
    target 11
    bw 7
  ]
  edge [
    source 11
    target 12
    bw 32
  ]
  edge [
    source 12
    target 13
    bw 4
  ]
  edge [
    source 13
    target 14
    bw 34
  ]
]
