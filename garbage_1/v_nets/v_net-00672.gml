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
  id 672
  arrival_time 14330.486245827396
  lifetime 1766.2977040904113
  num_nodes 14
  type "path"
  node [
    id 0
    label "0"
    cpu 31
    gpu 49
    rom 46
  ]
  node [
    id 1
    label "1"
    cpu 6
    gpu 40
    rom 5
  ]
  node [
    id 2
    label "2"
    cpu 23
    gpu 41
    rom 43
  ]
  node [
    id 3
    label "3"
    cpu 4
    gpu 49
    rom 31
  ]
  node [
    id 4
    label "4"
    cpu 36
    gpu 32
    rom 20
  ]
  node [
    id 5
    label "5"
    cpu 43
    gpu 39
    rom 23
  ]
  node [
    id 6
    label "6"
    cpu 40
    gpu 35
    rom 1
  ]
  node [
    id 7
    label "7"
    cpu 49
    gpu 22
    rom 39
  ]
  node [
    id 8
    label "8"
    cpu 42
    gpu 41
    rom 19
  ]
  node [
    id 9
    label "9"
    cpu 39
    gpu 25
    rom 29
  ]
  node [
    id 10
    label "10"
    cpu 4
    gpu 19
    rom 29
  ]
  node [
    id 11
    label "11"
    cpu 10
    gpu 41
    rom 29
  ]
  node [
    id 12
    label "12"
    cpu 37
    gpu 40
    rom 24
  ]
  node [
    id 13
    label "13"
    cpu 8
    gpu 13
    rom 43
  ]
  edge [
    source 0
    target 1
    bw 20
  ]
  edge [
    source 1
    target 2
    bw 32
  ]
  edge [
    source 2
    target 3
    bw 40
  ]
  edge [
    source 3
    target 4
    bw 25
  ]
  edge [
    source 4
    target 5
    bw 23
  ]
  edge [
    source 5
    target 6
    bw 50
  ]
  edge [
    source 6
    target 7
    bw 26
  ]
  edge [
    source 7
    target 8
    bw 4
  ]
  edge [
    source 8
    target 9
    bw 11
  ]
  edge [
    source 9
    target 10
    bw 5
  ]
  edge [
    source 10
    target 11
    bw 38
  ]
  edge [
    source 11
    target 12
    bw 39
  ]
  edge [
    source 12
    target 13
    bw 16
  ]
]
